"""Alternate partial CTA: adjacent query heads share each logical K/V load."""


def grouped_partial(defines, b, r, scale, mask_batch, mask):
    return (
        defines
        + f"""constexpr int R={r};
__global__ void decode_group(const T* q,const T* k,const T* v,const float* bias,void* dst){{
 const int i=threadIdx.x,l=i&31,w=i>>5,h=blockIdx.y*R,b=blockIdx.z,chunk=blockIdx.x;
 const int begin=chunk*Tile,count=min(Tile,S-begin),kv=((b*G+h/(H/G))*S)*D,qi=(b*H+h)*D;
 __shared__ float scores[R][Tile],rowmax[R],mass[R];float qreg[R][D/32];
 #pragma unroll
 for(int j=0;j<D/32;++j){{
 #pragma unroll
 for(int u=0;u<R;++u)qreg[u][j]=ld(q[qi+u*D+j*32+l]);}}
 for(int t=w;t<count;t+=NT/32){{int key=begin+t;float dot[R]={{}};
   #pragma unroll
   for(int j=0;j<D/32;++j){{float kval=ld(k[kv+key*D+j*32+l]);
   #pragma unroll
   for(int u=0;u<R;++u)dot[u]=fmaf(qreg[u][j],kval,dot[u]);}}
   #pragma unroll
   for(int u=0;u<R;++u){{float x=sumwarp(dot[u]);if(l==0)scores[u][t]=__fmul_rn(x,{float(scale).hex()}f){mask};}}
 }}
 __syncthreads();
 if(w<R){{float mx=-INFINITY;for(int t=l;t<count;t+=32)mx=fmaxf(mx,scores[w][t]);
 #pragma unroll
 for(int o=16;o;o>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffffu,mx,o));
 if(!l)rowmax[w]=mx;}}
 __syncthreads();
 for(int t=i;t<count;t+=NT){{
 #pragma unroll
 for(int u=0;u<R;++u) scores[u][t]=(rowmax[u]==-INFINITY ? 0.f : expf(scores[u][t]-rowmax[u]));}}
 __syncthreads();
 if(w<R){{float sm=0.f;for(int t=l;t<count;t+=32)sm+=scores[w][t];sm=sumwarp(sm);if(!l)mass[w]=sm;}}
 float acc[R]={{}};if(i<D){{for(int t=0;t<count;++t){{float val=ld(v[kv+(begin+t)*D+i]);
 #pragma unroll
 for(int u=0;u<R;++u)acc[u]=fmaf(scores[u][t],val,acc[u]);}}}}
 __syncthreads();
 #pragma unroll
 for(int u=0;u<R;u++){{ const int base=(((b*H+h+u)*C)+chunk)*(D+2);
 if constexpr(C==1){{if(i<D)((T*)dst)[qi+u*D+i]=rt(mass[u]==0.f ? 0.f : acc[u]/mass[u]);}}
 else{{if(i<D)((float*)dst)[base+i]=acc[u];if(!i){{((float*)dst)[base+D]=rowmax[u];((float*)dst)[base+D+1]=mass[u];}}}}
 }}
}}
extern "C" int launch(void** p,int,void* stream){{decode_group<<<dim3(C,H/R,{b}),NT,0,(cudaStream_t)stream>>>((T*)p[0],(T*)p[1],(T*)p[2],{("(float*)p[3]" if mask_batch else "nullptr")},p[{4 if mask_batch else 3}]);return int(cudaGetLastError());}}
"""
    )
