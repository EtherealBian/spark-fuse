// Host ABI for one functional pair of variable-length K/V concatenations.
#include <ATen/Functions.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/pybind11.h>
#include <array>
#include <cstdint>
namespace py=pybind11;
namespace {
using Launch=int (*)(void**,int,void*);
class AppendIO {
 public:
  AppendIO(py::object owner,uintptr_t entry,int device,int lo,int hi,int b,int g,int d,bool bf16)
    : owner_(std::move(owner)),launch_(reinterpret_cast<Launch>(entry)),device_(device),lo_(lo),hi_(hi),b_(b),g_(g),d_(d),dt_(bf16?at::kBFloat16:at::kHalf){
    if(!entry || device<0 || device>127 || lo<1 || lo>=hi || hi>=32768 || b<1 || b>32 || g<1 || g>256 || (d!=64 && d!=128) || int64_t(b)*g*(hi+1)*d>=(1LL<<29))throw py::value_error("invalid append plan");
  }
  py::tuple run(const py::args& args)const{
    if(args.size()!=4 || !c10::InferenceMode::is_enabled())throw py::value_error("four inference tensors required");
    std::array<at::Tensor,4> x;std::array<void*,6> ptr;int past=0,flags=0;bool supported=true;
    for(int i=0;i<4;++i){
      if(reinterpret_cast<PyObject*>(Py_TYPE(args[i].ptr()))!=THPVariableClass)throw py::value_error("expected exact Tensor");
      x[i]=THPVariable_Unpack(args[i].ptr());const auto& t=x[i];
      if(!t.is_cuda() || t.get_device()!=device_ || t.scalar_type()!=dt_ || t.layout()!=at::kStrided || t.is_neg() || t.is_conj() || t.dim()!=4 || t.size(0)!=b_ || t.size(1)!=g_ || t.size(3)!=d_)throw py::value_error("invalid append input");
      if(i==0){if(t.size(2)<lo_ || t.size(2)>hi_)throw py::value_error("past outside export bounds");past=static_cast<int>(t.size(2));}
      if(t.size(2)!=(i%2?1:past))throw py::value_error("cache and token lengths do not agree");
      if(!t.is_contiguous()){
        if(i%2==0 && t.stride(3)==1 && t.stride(1)==d_ && t.stride(2)==int64_t(g_)*d_ && (b_==1 || t.stride(0)==int64_t(past)*g_*d_))flags|=1<<(i/2);
        else supported=false;
      }
      ptr[i]=t.data_ptr();if(!ptr[i])throw py::value_error("null append storage");
    }
    c10::cuda::CUDAGuard guard(device_);auto stream=c10::cuda::getCurrentCUDAStream(device_);
    for(auto& t:x)c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(),stream);
    at::Tensor k,v;
    if(!supported){k=at::cat({x[0],x[1]},2);v=at::cat({x[2],x[3]},2);}else{
      k=at::empty({b_,g_,past+1,d_},x[0].options());v=at::empty_like(k);
      ptr[4]=k.data_ptr();ptr[5]=v.data_ptr();int ec=launch_(ptr.data(),(past<<2)|flags,reinterpret_cast<void*>(stream.stream()));
      if(ec)throw std::runtime_error("paired append CUDA returned "+std::to_string(ec));
    }
    py::tuple out(2);out[0]=py::reinterpret_steal<py::object>(THPVariable_Wrap(k));out[1]=py::reinterpret_steal<py::object>(THPVariable_Wrap(v));return out;
  }
 private:
  py::object owner_;Launch launch_;c10::DeviceIndex device_;int lo_,hi_,b_,g_,d_;at::ScalarType dt_;
};
}
void bind_append_io(py::module_& m){py::class_<AppendIO>(m,"_AppendIO").def(py::init<py::object,uintptr_t,int,int,int,int,int,int,bool>()).def("run",&AppendIO::run);}
