// Tensor validation, fresh output allocation and stream-aware dispatch for trusted generated kernels.
#include <ATen/Functions.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstdint>

namespace py = pybind11;
using Shape = std::vector<int64_t>;
using Launch = int (*)(void**, int, void*);

static at::ScalarType dtype(int key) {
  switch (key) {
    case 0: return at::kFloat;
    case 1: return at::kHalf;
    case 2: return at::kBFloat16;
    default: throw py::value_error("unsupported dtype");
  }
}

class HostPlan {
 public:
  HostPlan(py::object owner, uintptr_t entry, int device, int rows, size_t nin,
           std::vector<Shape> shapes, std::vector<int> types, bool leaf)
      : owner_(std::move(owner)), launch_(reinterpret_cast<Launch>(entry)),
        device_(device), rows_(rows), nin_(nin), shapes_(std::move(shapes)), leaf_(leaf) {
    if (!entry || rows<=0 || !nin || shapes_.size()<=nin || shapes_.size()!=types.size() ||
        (leaf && shapes_.size()!=nin+1)) throw py::value_error("invalid plan descriptor");
    for(int k:types) types_.push_back(dtype(k));
  }

  py::object run(const py::args &args) const {
    if(args.size()!=nin_) throw py::value_error("incorrect tensor count");
    c10::cuda::CUDAGuard guard(device_);
    auto stream=c10::cuda::getCurrentCUDAStream(device_);
    c10::SmallVector<void*, 12> ptrs;
    for(size_t i=0;i<nin_;++i){
      if (!THPVariable_Check(args[i].ptr())) throw py::value_error("tensor expected");
      const at::Tensor &t=THPVariable_Unpack(args[i].ptr());
      if (!t.is_cuda() || t.get_device()!=device_ || t.scalar_type()!=types_[i] ||
          t.layout()!=at::kStrided || !t.is_contiguous() || !t.sizes().equals(shapes_[i]) ||
          t.is_neg() || t.is_conj()) throw py::value_error("input shape/dtype/device/layout mismatch");
      if (at::GradMode::is_enabled() && t.requires_grad()) throw py::value_error("inference only");
      ptrs.push_back(t.data_ptr());
      c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(),stream);
    }
    c10::SmallVector<at::Tensor, 4> out;
    for (size_t i=nin_;i<shapes_.size();++i){
      out.push_back(at::empty(shapes_[i],at::TensorOptions().device(at::Device(at::kCUDA,device_)).dtype(types_[i])));
      ptrs.push_back(out.back().data_ptr());
    }
    const int ec=launch_(ptrs.data(),rows_,reinterpret_cast<void*>(stream.stream()));
    if(ec) throw std::runtime_error("CUDA launch returned "+std::to_string(ec));
    if(leaf_) return py::reinterpret_steal<py::object>(THPVariable_Wrap(out[0]));
    py::tuple result(out.size());
    for(size_t i=0;i<out.size();++i) result[i]=py::reinterpret_steal<py::object>(THPVariable_Wrap(out[i]));
    return result;
  }
 private:
  py::object owner_; // Keeps the kernel shared object loaded, even if the compiler object dies.
  Launch launch_;
  c10::DeviceIndex device_;
  int rows_;
  size_t nin_;
  std::vector<Shape> shapes_;
  std::vector<at::ScalarType> types_;
  bool leaf_;
};
PYBIND11_MODULE(_sparkfuse_host,m){
  py::class_<HostPlan>(m,"_Plan")
    .def(py::init<py::object, uintptr_t, int, int, size_t, std::vector<Shape>,std::vector<int>,bool>())
    .def("run",&HostPlan::run);
}
