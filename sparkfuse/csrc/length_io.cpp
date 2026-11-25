// A single native host call validates all variable-length descriptors and owns both launches.
// Kernel machine code and numerical policy are unchanged from the ctypes implementation.
#include <ATen/Functions.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/pybind11.h>
#include <array>
#include <cstdint>

namespace py = pybind11;
namespace {
using Launch = int (*)(void**, int, void*);
using Four = std::array<int64_t,4>;

class LengthIO {
 public:
  LengthIO(py::object owner, uintptr_t a, uintptr_t m, int device, int lo, int hi,
           int b, int h, int g, int d, int mask, int tile, bool bf16)
      : owner_(std::move(owner)), a_(reinterpret_cast<Launch>(a)), m_(reinterpret_cast<Launch>(m)),
        device_(device),lo_(lo),hi_(hi),b_(b),h_(h),g_(g),d_(d),mask_(mask),tile_(tile),
        dt_(bf16 ? at::kBFloat16 : at::kHalf) {
    if (!a || device<0 || device>127 || lo<2 || lo>=hi || hi>32768 || b<1 || b>32 ||
        g<1 || h<g || h>256 || h%g || (d!=64 && d!=128) ||
        (mask && mask!=1 && mask!=b) || (tile!=32 && tile!=64 && tile!=128 && tile!=256) ||
        bool(m)!=(hi>tile) || int64_t(b)*g*hi*d>=(1LL<<29) ||
        int64_t(b)*h*((hi+tile-1)/tile)*(d+2)>=(1LL<<29))
      throw py::value_error("invalid length IO descriptor");
  }

  py::object run(const py::args& args) const {
    const size_t n=mask_ ? 4 : 3;
    if (args.size()!=n || !c10::InferenceMode::is_enabled()) throw py::value_error("inference tensor arguments required");
    for (const auto& x:args) if (reinterpret_cast<PyObject*>(Py_TYPE(x.ptr()))!=THPVariableClass)
      throw py::value_error("expected exact Tensor arguments");
    const auto& key=THPVariable_Unpack(args[1].ptr());
    if(key.dim()!=4 || key.size(2)<lo_ || key.size(2)>hi_) throw py::value_error("invalid KV length");
    const int s=static_cast<int>(key.size(2)), c=(s+tile_-1)/tile_;
    const Four q={b_,h_,1,d_},kv={b_,g_,s,d_},bias={mask_,1,1,s};
    std::array<void*,5> ptr{};
    c10::cuda::CUDAGuard guard(device_);
    auto stream=c10::cuda::getCurrentCUDAStream(device_);
    for(size_t i=0;i<n;++i){
      const auto& t=THPVariable_Unpack(args[i].ptr());
      const auto& sh=i==0?q:(i==3?bias:kv);
      if(!t.is_cuda() || t.get_device()!=device_ || t.scalar_type()!=dt_ ||
         t.layout()!=at::kStrided || !t.sizes().equals(sh) || t.is_neg() || t.is_conj())
        throw py::value_error("input shape/dtype/device/layout mismatch");
      int64_t stride=1;
      for(int j=3;j>=0;--j){
        if(t.stride(j)!=stride)throw py::value_error("non-canonical input stride");
        stride*=sh[j];
      }
      ptr[i]=t.data_ptr();
      if(!ptr[i] || reinterpret_cast<uintptr_t>(ptr[i])%16)throw py::value_error("unaligned or missing input storage");
    }
    for(const auto& x:args)c10::cuda::CUDACachingAllocator::recordStream(THPVariable_Unpack(x.ptr()).storage().data_ptr(),stream);
    const auto opts=at::TensorOptions().device(at::Device(at::kCUDA,device_)).dtype(dt_);
    at::Tensor y=at::empty(q,opts), partial;
    if(c==1)ptr[n]=y.data_ptr();
    else{partial=at::empty({b_,h_,c,d_+2},opts.dtype(at::kFloat));ptr[n]=partial.data_ptr();}
    // Allocation and both launches are on this same current stream. Only cross-stream inputs
    // need recording. The caller owns the fresh result; private partials never escape.
    void* current=reinterpret_cast<void*>(stream.stream());
    int ec=a_(ptr.data(),s,current);
    if(!ec && c!=1){void* merge[]={ptr[n],y.data_ptr()};ec=m_(merge,s,current);}
    if(ec)throw std::runtime_error("length decode CUDA launch returned "+std::to_string(ec));
    return py::reinterpret_steal<py::object>(THPVariable_Wrap(y));
  }
 private:
  py::object owner_;
  Launch a_,m_;
  c10::DeviceIndex device_;
  int lo_,hi_,b_,h_,g_,d_,mask_,tile_;
  at::ScalarType dt_;
};
}
void bind_length_io(py::module_& m){
  py::class_<LengthIO>(m,"_LengthIO")
    .def(py::init<py::object,uintptr_t,uintptr_t,int,int,int,int,int,int,int,int,int,bool>())
    .def("run",&LengthIO::run);
}
