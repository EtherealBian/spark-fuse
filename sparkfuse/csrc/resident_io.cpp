// Whole-storage staging and independent output clones for a trusted, qualified CUDA graph.
// Never cache caller tensors or their addresses; validate *all* arguments before the first write.
#include <ATen/Functions.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>

namespace py = pybind11;
namespace {
using Shape = std::vector<int64_t>;
using SlotLaunch = int (*)(void**, int, void*);
struct BadStorage : std::runtime_error { using std::runtime_error::runtime_error; };
struct BadLayout : std::runtime_error { using std::runtime_error::runtime_error; };

struct Span {
  uintptr_t addr;
  size_t bytes;
  bool overlaps(const Span& b) const {
    return bytes && b.bytes && (addr <= b.addr ? b.addr - addr < bytes : addr - b.addr < b.bytes);
  }
};
Span span(const at::Tensor& t) {
  return {reinterpret_cast<uintptr_t>(t.storage().data_ptr().get()), t.storage().nbytes()};
}

class ResidentIO {
 public:
  ResidentIO(py::tuple staged, py::tuple outputs, py::object replay,
             std::vector<Shape> strides, py::object plain, py::object parameter, py::tuple protected_tensors,
             py::object slot_lib, uintptr_t slot_entry, bool cuda_slots)
      : replay_(std::move(replay)), plain_(std::move(plain)), parameter_(std::move(parameter)),
        slot_lib_(std::move(slot_lib)), slot_launch_(reinterpret_cast<SlotLaunch>(slot_entry)),
        cuda_slots_(cuda_slots), strides_(std::move(strides)) {
    if (bool(slot_launch_) == slot_lib_.is_none() || (cuda_slots_ && !slot_launch_) ||
        (slot_launch_ && outputs.size() > 32))
      throw py::value_error("invalid slot-copy binding");
    if (!staged.size() || !outputs.size() || strides_.size() != staged.size() ||
        !PyCallable_Check(replay_.ptr())) throw py::value_error("invalid resident bindings");
    for (auto x : staged) inputs_.push_back(binding(x));
    for (auto x : outputs) outputs_.push_back(binding(x));
    device_ = inputs_[0].get_device();
    for (size_t i = 0; i < inputs_.size(); ++i) {
      if (inputs_[i].get_device() != device_ || strides_[i].size() != static_cast<size_t>(inputs_[i].dim()))
        throw py::value_error("invalid staging descriptor");
    }
    for (const auto& t : outputs_) if (t.get_device() != device_) throw py::value_error("mixed output devices");
    // Do not retain user weights or qualification requests beyond their ordinary lifetimes.
    // StorageImpl (not TensorImpl) also covers detached aliases. Read the live address on use.
    for (auto x : protected_tensors) {
      if (!THPVariable_Check(x.ptr())) throw py::value_error("invalid protected binding");
      const auto& t = THPVariable_Unpack(x.ptr());
      if (t.is_cuda() && t.get_device() == device_ && t.layout() == at::kStrided)
        protected_.push_back(t.storage().getWeakStorageImpl());
    }
    owner_ = c10::cuda::getCurrentCUDAStream(device_).id();
  }

  py::tuple run(const py::args& args) const {
    const auto stream = validate_inputs(args);
    stage_and_replay(args, stream);
    py::tuple result(outputs_.size());
    for (size_t i = 0; i < outputs_.size(); ++i)
      result[i] = py::reinterpret_steal<py::object>(THPVariable_Wrap(outputs_[i].clone()));
    return result;
  }

  // New tensors keep separate StorageImpls. Batch only the byte transfers, never the allocations.
  py::tuple run_fused(const py::args& args) const {
    const auto stream = validate_inputs(args);
    if (!slot_launch_) throw BadLayout("fresh CUDA transport was not prepared");
    py::tuple result(outputs_.size());
    void* ptr[64];
    for (size_t i = 0; i < outputs_.size(); ++i) {
      const auto& src = outputs_[i];
      // Captured ABI guarantees whole contiguous storage. Preserve singleton strides as well.
      auto dst = at::empty_strided(src.sizes(), src.strides(), src.options());
      ptr[i] = dst.data_ptr(); ptr[outputs_.size() + i] = src.data_ptr();
      result[i] = py::reinterpret_steal<py::object>(THPVariable_Wrap(std::move(dst)));
    }
    stage_and_replay(args, stream);
    // Newly allocated on the current owner stream; no cross-stream borrower to record here.
    C10_CUDA_CHECK(static_cast<cudaError_t>(slot_launch_(ptr, int(outputs_.size()), stream.stream())));
    return result;
  }

  void run_into(py::tuple args, py::tuple destinations) const {
    const auto stream = validate_inputs(args);
    if (destinations.size() != outputs_.size()) throw BadLayout("output buffer arity changed");
    c10::SmallVector<Span, 32> forbidden;
    for (const auto& t : inputs_) forbidden.push_back(span(t));
    for (const auto& t : outputs_) forbidden.push_back(span(t));
    for (auto h : args) {
      const auto current = span(THPVariable_Unpack(h.ptr()));
      for (const auto& s : forbidden) if (s.overlaps(current)) throw BadStorage("request overlaps resident IO storage");
      forbidden.push_back(current);
    }
    for (const auto& ref : protected_) if (auto st = ref.lock())
      forbidden.push_back({reinterpret_cast<uintptr_t>(st->data_ptr().get()), st->nbytes()});
    for (size_t i = 0; i < outputs_.size(); ++i) {
      const auto h = destinations[i];
      if (reinterpret_cast<PyObject*>(Py_TYPE(h.ptr())) != plain_.ptr())
        throw BadStorage("output buffer must be a plain non-gradient CUDA tensor");
      const auto& t = THPVariable_Unpack(h.ptr());
      const auto& base = outputs_[i];
      if (t.requires_grad() || t.device() != base.device() || t.layout() != at::kStrided ||
          t.is_neg() || t.is_conj() || !t.is_contiguous() || t.numel() <= 0 || t.storage_offset() ||
          t.storage().nbytes() != t.nbytes()) throw BadStorage("output buffer requires whole contiguous storage");
      if (t.scalar_type() != base.scalar_type() || !t.sizes().equals(base.sizes()) ||
          !t.strides().equals(base.strides())) throw BadLayout("output buffer layout or dtype changed");
      const Span current = span(t);
      for (const auto& s : forbidden) if (s.overlaps(current)) throw BadStorage("output buffer overlaps protected storage");
      forbidden.push_back(current);
    }
    stage_and_replay(args, stream);
    if (cuda_slots_) {
      // The generated schedule was specialized to these private outputs' byte counts.
      // Only caller addresses vary; all destination spans were validated before staging.
      void* ptr[64];
      for (size_t i = 0; i < outputs_.size(); ++i) {
        const auto& t = THPVariable_Unpack(destinations[i].ptr());
        c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(), stream);
        ptr[i] = t.data_ptr(); ptr[outputs_.size() + i] = outputs_[i].data_ptr();
      }
      C10_CUDA_CHECK(static_cast<cudaError_t>(slot_launch_(ptr, int(outputs_.size()), stream.stream())));
      // ATen copy_ does this implicitly; raw byte stores must invalidate regular tensor versions too.
      for (auto h : destinations) THPVariable_Unpack(h.ptr()).unsafeGetTensorImpl()->bump_version();
      return;
    }
    for (size_t i = 0; i < outputs_.size(); ++i) {
      const auto& t = THPVariable_Unpack(destinations[i].ptr());
      c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(), stream);
      t.copy_(outputs_[i]);
    }
  }
 private:
  c10::cuda::CUDAStream validate_inputs(const py::tuple& args) const {
    if (args.size() != inputs_.size()) throw BadLayout("resident request arity changed");
    const auto stream = c10::cuda::getCurrentCUDAStream(device_);
    if (!c10::InferenceMode::is_enabled() || stream.id() != owner_ || c10::cuda::current_device() != device_)
      throw BadLayout("requires inference mode on the resident owner stream and device");
    c10::SmallVector<Span, 8> stores;
    for (size_t i = 0; i < inputs_.size(); ++i) {
      const auto h = args[i];
      const auto* cl = reinterpret_cast<PyObject*>(Py_TYPE(h.ptr()));
      if ((cl != plain_.ptr() && cl != parameter_.ptr()) || !THPVariable_Check(h.ptr()))
        throw BadStorage("plain whole CUDA tensors required");
      const auto& t = THPVariable_Unpack(h.ptr());
      const auto& base = inputs_[i];
      if (t.device() != base.device() || t.layout() != at::kStrided || t.is_neg() || t.is_conj() ||
          !t.is_contiguous() || t.numel() <= 0 || t.storage_offset() || t.storage().nbytes() != t.nbytes())
        throw BadStorage("request requires whole contiguous tensors on the capture device");
      if (!t.sizes().equals(base.sizes()) || !t.strides().equals(strides_[i]) || t.scalar_type() != base.scalar_type())
        throw BadLayout("resident request layout or precision mode changed");
      const auto p = span(t);
      for (auto b : stores) if (p.overlaps(b)) throw BadStorage("request inputs share storage");
      stores.push_back(p);
    }
    return stream;
  }
  void stage_and_replay(const py::tuple& args, c10::cuda::CUDAStream stream) const {
    // From here a backend error must invalidate the handle. Public contract errors above are recoverable.
    for (size_t i = 0; i < inputs_.size(); ++i) {
      const auto& t = THPVariable_Unpack(args[i].ptr());
      c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(), stream);
      inputs_[i].copy_(t);
    }
    replay_();
  }
  static at::Tensor binding(py::handle x) {
    if (!THPVariable_Check(x.ptr())) throw py::value_error("expected tensor binding");
    const auto& t = THPVariable_Unpack(x.ptr());
    if (!t.is_cuda() || t.layout() != at::kStrided || !t.is_contiguous() || t.numel() <= 0 ||
        t.storage_offset() || t.storage().nbytes() != t.nbytes() || t.is_neg() || t.is_conj())
      throw py::value_error("invalid resident owner tensor");
    return t;
  }
  py::object replay_, plain_, parameter_, slot_lib_;
  SlotLaunch slot_launch_;
  bool cuda_slots_;
  std::vector<Shape> strides_;
  std::vector<at::Tensor> inputs_, outputs_;
  std::vector<c10::weak_intrusive_ptr<c10::StorageImpl>> protected_;
  c10::DeviceIndex device_;
  c10::StreamId owner_;
};
} // namespace

void bind_resident_io(py::module_& m) {
  py::register_exception<BadStorage>(m,"_ResidentBadStorage");
  py::register_exception<BadLayout>(m,"_ResidentBadLayout");
  py::class_<ResidentIO>(m,"_ResidentIO")
    .def(py::init<py::tuple,py::tuple,py::object,std::vector<Shape>,py::object,py::object,py::tuple,py::object,uintptr_t,bool>())
    .def("run", &ResidentIO::run)
    .def("run_fused", &ResidentIO::run_fused)
    .def("run_into", &ResidentIO::run_into);
}
