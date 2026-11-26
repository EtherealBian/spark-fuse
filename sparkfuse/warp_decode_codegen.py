"""Warp-private split-KV partials. Eight four-lane crews each score one token per round."""


def warp_partial(
    defines, b, h, g, d, upper, tile, scale, mask_batch, warps, dtype, *, page_pool=0, ragged=False
):
    bias = ("+ld(bias[" + ("batch*S+" if mask_batch > 1 else "") + "key])") if mask_batch else ""
    cv = "__ushort_as_bfloat16" if dtype == "bf16" else "__ushort_as_half"
    # In paged mode one split equals one logical page. Saturation is present in the
    # matched ATen graph; page numbers never become unchecked device addresses.
    table_arg = ",const int* pages" if page_pool else ""
    table_ptr = ",(const int*)p[3]" if page_pool else ""
    origin = (
        f"(max(0,min({page_pool - 1},pages[batch*C+chunk]))*G+h/(H/G))*Tile*D"
        if page_pool
        else "((batch*G+h/(H/G))*S)*D"
    )
    offset = "(key-begin)*D" if page_pool else "key*D"
    length_arg = "int" if page_pool else "int S"
    mask_ptr = f"(const T*)p[{3 + bool(page_pool)}]" if mask_batch else "nullptr"
    output = 3 + int(bool(page_pool)) + int(bool(mask_batch))
    if ragged:
        table_arg += ",const int* lengths"
        table_ptr += ",(const int*)p[4]"
        mask_ptr = "(const T*)p[5]"
        output = 6
    valid = "live" if ragged else "S"
    preflight = (
        "const int live=max(0,min(S,lengths[batch])); if(begin>=live){if(C==1){for(int j=0;j<D/32;++j)((T*)dst)[(batch*H+h)*D+j*32+lane]=rt(0.f);}return;}"
        if ragged
        else ""
    )
    kv_decl = (";" + preflight + "const int kv=") if ragged else ",kv="
    return (
        defines
        + f"""__device__ __forceinline__ float cv(unsigned u){{return ld({cv}(static_cast<unsigned short>(u)));}}
// Aligned 16-byte packets are guaranteed by LengthIO; no warp communicates via shared memory.
__global__ void warp_partial(const T* q,const T* k,const T* v{table_arg},const T* bias,void* dst,int S){{
 constexpr int N=Tile/8;const int lane=threadIdx.x&31,crew=lane>>2,slice=lane&3;
 const int h=blockIdx.y*{warps}+(threadIdx.x>>5),batch=blockIdx.z,chunk=blockIdx.x;
 if(h>=H)return;const int C=(S+Tile-1)/Tile,begin=chunk*Tile{kv_decl}{origin},qi=(batch*H+h)*D;
 float qreg[D/32][8];
 #pragma unroll
 for(int j=0;j<D/32;++j){{uint4 z=*reinterpret_cast<const uint4*>(q+qi+j*32+slice*8);unsigned a[4]={{z.x,z.y,z.z,z.w}};
 #pragma unroll
 for(int t=0;t<4;t++){{qreg[j][2*t]=cv(a[t]);qreg[j][2*t+1]=cv(a[t]>>16);}}}}
 float scores[N],mx=-INFINITY;
 #pragma unroll
 for(int r=0;r<N;r++){{int key=begin+r*8+crew;float dot=0.f;
   if(key<{valid}){{
   #pragma unroll
   for(int j=0;j<D/32;++j){{uint4 z=*reinterpret_cast<const uint4*>(k+kv+{offset}+j*32+slice*8);unsigned a[4]={{z.x,z.y,z.z,z.w}};
   #pragma unroll
   for(int t=0;t<4;t++){{dot=fmaf(qreg[j][2*t],cv(a[t]),dot);dot=fmaf(qreg[j][2*t+1],cv(a[t]>>16),dot);}}}}}}
   dot+=__shfl_xor_sync(0xffffffffu,dot,2,4);dot+=__shfl_xor_sync(0xffffffffu,dot,1,4);
   float score=key<{valid} ? __fmul_rn(dot,{float(scale).hex()}f){bias} : -INFINITY;scores[r]=score;mx=fmaxf(mx,score);
 }}
 #pragma unroll
 for(int x=4;x<=16;x*=2)mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,x));
 float mass=0.f,acc[D/32]={{}};
 #pragma unroll
 for(int r=0;r<N;++r){{float weight=mx==-INFINITY ? 0.f : expf(scores[r]-mx);mass+=weight;
 #pragma unroll
 for(int u=0;u<8;++u){{int key=begin+r*8+u;float wt=__shfl_sync(0xffffffffu,weight,4*u);
 if(key<{valid}){{
 #pragma unroll
 for(int j=0;j<D/32;++j)acc[j]=fmaf(wt,ld(v[kv+{offset}+j*32+lane]),acc[j]);}}}}}}
 #pragma unroll
 for(int x=4;x<=16;x*=2)mass+=__shfl_xor_sync(0xffffffffu,mass,x);
 int base=((batch*H+h)*C+chunk)*(D+2);
 if(C==1){{
 #pragma unroll
 for(int j=0;j<D/32;++j)((T*)dst)[qi+j*32+lane]=rt(mass==0.f?0.f:acc[j]/mass);
 }}else{{
 #pragma unroll
 for(int j=0;j<D/32;++j)((float*)dst)[base+j*32+lane]=acc[j];
 if(!lane){{((float*)dst)[base+D]=mx;((float*)dst)[base+D+1]=mass;}}}}
}}
extern "C" int launch(void** p,{length_arg},void* stream){{if(S<1||S>{upper})return int(cudaErrorInvalidValue);int C=(S+Tile-1)/Tile;
 warp_partial<<<dim3(C,{(h + warps - 1) // warps},{b}),{32 * warps},0,(cudaStream_t)stream>>>((const T*)p[0],(const T*)p[1],(const T*)p[2]{table_ptr},{mask_ptr},p[{output}],S);return int(cudaGetLastError());}}
"""
    )
