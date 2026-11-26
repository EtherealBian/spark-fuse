"""Merge only initialized logical pages; empty rows must neither read scratch nor return NaN."""


def ragged_merge(defines, b, h):
    return (
        defines
        + f"""__global__ void ragged_merge(const float* partial,const int* lengths,T* out){{
 const int bh=blockIdx.x,i=threadIdx.x,live=max(0,min(S,lengths[bh/H])),used=(live+Tile-1)/Tile,stride=D+2;
 if(!used){{if(i<D)out[bh*D+i]=rt(0.f);return;}}
 const float* p=partial+bh*C*stride;__shared__ float weights[C],rowmax,total;
 if(i<32){{float mx=-INFINITY;for(int c=i;c<used;c+=32)mx=fmaxf(mx,p[c*stride+D]);
 #pragma unroll
 for(int o=16;o;o>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffffu,mx,o));if(!i)rowmax=mx;}}
 __syncthreads();for(int c=i;c<used;c+=NT)weights[c]=rowmax==-INFINITY?0.f:expf(p[c*stride+D]-rowmax);
 __syncthreads();if(i<32){{float sm=0.f;for(int c=i;c<used;c+=32)sm=fmaf(p[c*stride+D+1],weights[c],sm);sm=sumwarp(sm);if(!i)total=sm;}}
 float y=0.f;if(i<D)for(int c=0;c<used;++c)y=fmaf(p[c*stride+i],weights[c],y);
 __syncthreads();if(i<D)out[bh*D+i]=rt(total==0.f?0.f:y/total);
}}
extern "C" int launch(void** p,int,void* stream){{ragged_merge<<<{b * h},NT,0,(cudaStream_t)stream>>>((const float*)p[0],(const int*)p[1],(T*)p[2]);return int(cudaGetLastError());}}
"""
    )
