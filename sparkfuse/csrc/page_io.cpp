// One host boundary for read-only page decode. No shared output or persistent scratch.
#include <ATen/Functions.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/pybind11.h>
#include <array>

namespace py=pybind11;
namespace {
using Launch=int(*)(void**,int,void*);
class PageIO {
 public:
  PageIO(py::object owner, uintptr_t first, uintptr_t merge, int device,
         int batch,int heads,int groups,int width,int physical,int page,int seq,int mask,bool ragged,bool bf16)
    : owner_(std::move(owner)), first_(reinterpret_cast<Launch>(first)), merge_(reinterpret_cast<Launch>(merge)), device_(device),
      b_(batch),h_(heads),g_(groups),d_(width),p_(physical),t_(page),s_(seq),mask_(mask),ragged_(ragged),dt_(bf16?at::kBFloat16:at::kHalf) {
    if (!first || device<0 || device>127 || batch<1 || batch>32 || groups<1 || heads<groups || heads>256 || heads%groups ||
        (width!=64 && width!=128) || physical<1 || (page!=32 && page!=64) || seq<1 || seq>32768 ||
        (mask && mask!=1 && mask!=batch) || (ragged && !mask) || bool(merge)!=(seq>page) ||
        int64_t(physical)*groups*page*width >= (1LL<<29)) throw py::value_error("invalid paged IO geometry");
    c_=(seq+page-1)/page;
    if(int64_t(batch)*heads*c_*(width+2)>=(1LL<<29))throw py::value_error("invalid paged workspace geometry");
  }
  py::object run(const py::args& args) const {
    const size_t n=4+int(ragged_)+int(mask_!=0);
    if(args.size()!=n || !c10::InferenceMode::is_enabled())throw py::value_error("inference tensor arguments required");
    std::array<void*,7> ptrs{};
    const std::array<int64_t,4> q={b_,h_,1,d_}, bank={p_,g_,t_,d_}, bias={mask_,1,1,s_};
    const std::array<int64_t,2> table={b_,c_}; const std::array<int64_t,1> length={b_};
    c10::cuda::CUDAGuard guard(device_);auto stream=c10::cuda::getCurrentCUDAStream(device_);
    for(size_t i=0;i<n;++i){
      if(reinterpret_cast<PyObject*>(Py_TYPE(args[i].ptr()))!=THPVariableClass) throw py::value_error("expected exact Tensor");
      const auto& x=THPVariable_Unpack(args[i].ptr()); bool integer=i==3 || (ragged_ && i==4);
      at::IntArrayRef shape=i==0?at::IntArrayRef(q):i<3?at::IntArrayRef(bank):i==3?at::IntArrayRef(table):integer?at::IntArrayRef(length):at::IntArrayRef(bias);
      if(!x.is_cuda() || x.get_device()!=device_ || x.scalar_type()!=(integer?at::kInt:dt_) || x.layout()!=at::kStrided ||
          !x.sizes().equals(shape) || !x.is_contiguous() || x.is_neg() || x.is_conj())throw py::value_error("invalid page operand descriptor");
      ptrs[i]=x.data_ptr(); if(!ptrs[i] || reinterpret_cast<uintptr_t>(ptrs[i])%(integer?4:16))throw py::value_error("invalid page operand alignment");
    }
    for(auto x:args)c10::cuda::CUDACachingAllocator::recordStream(THPVariable_Unpack(x.ptr()).storage().data_ptr(),stream);
    const auto opts=at::TensorOptions().device(at::Device(at::kCUDA,device_)).dtype(dt_);
    at::Tensor out=at::empty(q,opts),partial;
    if(c_>1){partial=at::empty({b_,h_,c_,d_+2},opts.dtype(at::kFloat));ptrs[n]=partial.data_ptr();}else ptrs[n]=out.data_ptr();
    void* current=reinterpret_cast<void*>(stream.stream());int ec=first_(ptrs.data(),1,current);
    if(!ec && merge_){std::array<void*,3> m={ptrs[n],ragged_?ptrs[4]:out.data_ptr(),out.data_ptr()};ec=merge_(m.data(),1,current);}
    if(ec)throw std::runtime_error("paged CUDA launch returned "+std::to_string(ec));
    return py::reinterpret_steal<py::object>(THPVariable_Wrap(out));
  }
 private:
  py::object owner_; Launch first_,merge_;c10::DeviceIndex device_;int b_,h_,g_,d_,p_,t_,s_,mask_,c_;bool ragged_;at::ScalarType dt_;
};
}
void bind_page_io(py::module_& m){py::class_<PageIO>(m,"_PageIO")
  .def(py::init<py::object,uintptr_t,uintptr_t,int,int,int,int,int,int,int,int,int,bool,bool>()).def("run",&PageIO::run);}
