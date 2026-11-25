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
    case 3: return at::kChar;
    case 4: return at::kInt;
    default: throw py::value_error("unsupported dtype");
  }
}

class HostPlan {
 public:
  HostPlan(py::object owner, uintptr_t entry, int device, int rows, size_t nin,
           std::vector<Shape> shapes, std::vector<int> types, bool leaf, int lo, int hi,
           std::vector<Shape> strides)
      : owner_(std::move(owner)), launch_(reinterpret_cast<Launch>(entry)),
        device_(device), rows_(rows), nin_(nin), shapes_(std::move(shapes)), strides_(std::move(strides)), leaf_(leaf), lo_(lo), hi_(hi), row_input_(nin) {
    if (!entry || rows<=0 || !nin || shapes_.size()<=nin || shapes_.size()!=types.size() ||
        (leaf && shapes_.size()!=nin+1)) throw py::value_error("invalid plan descriptor");
    if (!strides_.empty()) {
      if (strides_.size()!=nin || lo || hi) throw py::value_error("invalid static stride descriptor");
      for (size_t i=0;i<nin;++i) {
        if (strides_[i].empty()) continue;
        if (strides_[i].size()!=shapes_[i].size()) throw py::value_error("invalid stride rank");
        for (auto step : strides_[i]) if (step<=0) throw py::value_error("non-positive stride");
      }
    }
    if ((lo || hi) && (lo < 1 || rows < lo || rows > hi))
      throw py::value_error("invalid dynamic bounds");
    for (size_t i=0; i<shapes_.size(); ++i) {
      auto &shape=shapes_[i];
      for (size_t j=0; j<shape.size(); ++j) {
        if (lo && j==0 && shape.size()==2 && shape[0]==-1) {
          if (i<row_input_) row_input_=i;
        } else if (shape[j]<=0) throw py::value_error("invalid shape descriptor");
      }
    }
    if (lo && row_input_==nin_) throw py::value_error("missing dynamic row input");
    for(int k:types) types_.push_back(dtype(k));
  }

  py::object run(const py::args &args) const {
    return lo_ ? run_impl<true>(args) : run_impl<false>(args);
  }

 private:
  template<bool Dynamic>
  py::object run_impl(const py::args &args) const {
    if(args.size()!=nin_) throw py::value_error("incorrect tensor count");
    int rows=rows_;
    if constexpr (Dynamic) {
      if (!THPVariable_Check(args[row_input_].ptr())) throw py::value_error("row tensor expected");
      const auto &t=THPVariable_Unpack(args[row_input_].ptr());
      if (t.dim()!=2) throw py::value_error("row input shape mismatch");
      const auto n=t.size(0);
      if (n<lo_ || n>hi_) throw py::value_error("row count outside export bounds");
      rows=static_cast<int>(n);
    }
    c10::cuda::CUDAGuard guard(device_);
    auto stream=c10::cuda::getCurrentCUDAStream(device_);
    c10::SmallVector<void*, 12> ptrs;
    for(size_t i=0;i<nin_;++i){
      if (!THPVariable_Check(args[i].ptr())) throw py::value_error("tensor expected");
      const at::Tensor &t=THPVariable_Unpack(args[i].ptr());
      const auto &shape=shapes_[i];
      const bool shape_ok=(Dynamic && !shape.empty() && shape[0]==-1)
        ? (t.dim()==2 && t.size(0)==rows && t.size(1)==shape[1]) : t.sizes().equals(shape);
      if (!t.is_cuda() || t.get_device()!=device_ || t.scalar_type()!=types_[i] ||
          t.layout()!=at::kStrided || !shape_ok ||
          ((strides_.empty() || strides_[i].empty()) ? !t.is_contiguous() : !t.strides().equals(strides_[i])) ||
          t.is_neg() || t.is_conj()) throw py::value_error("input shape/dtype/device/layout mismatch");
      if (at::GradMode::is_enabled() && t.requires_grad()) throw py::value_error("inference only");
      ptrs.push_back(t.data_ptr());
      c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(),stream);
    }
    c10::SmallVector<at::Tensor, 4> out;
    for (size_t i=nin_;i<shapes_.size();++i){
      const auto opts=at::TensorOptions().device(at::Device(at::kCUDA,device_)).dtype(types_[i]);
      const auto &shape=shapes_[i];
      if (Dynamic && !shape.empty() && shape[0]==-1)
        out.push_back(at::empty({rows,shape[1]},opts));
      else out.push_back(at::empty(shape,opts));
      ptrs.push_back(out.back().data_ptr());
    }
    const int ec=launch_(ptrs.data(),rows,reinterpret_cast<void*>(stream.stream()));
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
  std::vector<Shape> shapes_, strides_;
  std::vector<at::ScalarType> types_;
  bool leaf_;
  int lo_, hi_;
  size_t row_input_;
};
void bind_resident_io(py::module_&);
PYBIND11_MODULE(_sparkfuse_host,m){
  bind_resident_io(m);
  py::class_<HostPlan>(m,"_Plan")
    .def(py::init<py::object, uintptr_t, int, int, size_t, std::vector<Shape>,std::vector<int>,bool,int,int,std::vector<Shape>>())
    .def("run",&HostPlan::run);
}
