// Whole-storage staging and independent output clones for a trusted, qualified CUDA graph.
// Never cache caller tensors or their addresses; validate *all* arguments before the first write.
#include <ATen/Functions.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;
namespace {
using Shape = std::vector<int64_t>;
struct BadStorage : std::runtime_error { using std::runtime_error::runtime_error; };
struct BadLayout : std::runtime_error { using std::runtime_error::runtime_error; };

class ResidentIO {
 public:
  ResidentIO(py::tuple staged, py::tuple outputs, py::object replay,
             std::vector<Shape> strides, py::object plain, py::object parameter)
      : replay_(std::move(replay)), plain_(std::move(plain)), parameter_(std::move(parameter)),
        strides_(std::move(strides)) {
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
    owner_ = c10::cuda::getCurrentCUDAStream(device_).id();
  }

  py::tuple run(const py::args& args) const {
    if (args.size() != inputs_.size()) throw BadLayout("resident request arity changed");
    const auto stream = c10::cuda::getCurrentCUDAStream(device_);
    if (!c10::InferenceMode::is_enabled() || stream.id() != owner_ || c10::cuda::current_device() != device_)
      throw BadLayout("requires inference mode on the resident owner stream and device");
    c10::SmallVector<const void*, 8> stores;
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
      const auto* p = t.storage().data_ptr().get();
      for (auto b : stores) if (p == b) throw BadStorage("request inputs share storage");
      stores.push_back(p);
    }
    // From here a backend error must invalidate the handle. Public contract errors above are recoverable.
    for (size_t i = 0; i < inputs_.size(); ++i) {
      const auto& t = THPVariable_Unpack(args[i].ptr());
      c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(), stream);
      inputs_[i].copy_(t);
    }
    replay_();
    py::tuple result(outputs_.size());
    for (size_t i = 0; i < outputs_.size(); ++i)
      result[i] = py::reinterpret_steal<py::object>(THPVariable_Wrap(outputs_[i].clone()));
    return result;
  }
 private:
  static at::Tensor binding(py::handle x) {
    if (!THPVariable_Check(x.ptr())) throw py::value_error("expected tensor binding");
    const auto& t = THPVariable_Unpack(x.ptr());
    if (!t.is_cuda() || t.layout() != at::kStrided || !t.is_contiguous() || t.numel() <= 0 ||
        t.storage_offset() || t.storage().nbytes() != t.nbytes() || t.is_neg() || t.is_conj())
      throw py::value_error("invalid resident owner tensor");
    return t;
  }
  py::object replay_, plain_, parameter_;
  std::vector<Shape> strides_;
  std::vector<at::Tensor> inputs_, outputs_;
  c10::DeviceIndex device_;
  c10::StreamId owner_;
};
} // namespace

void bind_resident_io(py::module_& m) {
  py::register_exception<BadStorage>(m,"_ResidentBadStorage");
  py::register_exception<BadLayout>(m,"_ResidentBadLayout");
  py::class_<ResidentIO>(m,"_ResidentIO")
    .def(py::init<py::tuple,py::tuple,py::object,std::vector<Shape>,py::object,py::object>())
    .def("run", &ResidentIO::run);
}
