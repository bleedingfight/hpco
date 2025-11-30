import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
@cute.kernel
def naive_elementwise_kernel(c:cute.Tensor,a:cute.Tensor,b:cute.Tensor):
    tidx,_,_ = cute.arch.thread_idx()
    bidx,_,_ = cute.arch.block_idx()
    dimx,_,_ = cute.arch.block_dim()
    thread_idx = bidx*dimx+tidx
    m,n = a.shape
    mi = thread_idx//n
    ni = thread_idx%n
    c[mi,ni] = a[mi,ni]+b[mi,ni]
@cute.jit
def naive_elementwise_add(c:cute.Tensor,a:cute.Tensor,b:cute.Tensor):
    threads_per_block = 256
    m,n = a.shape
    kernel = naive_elementwise_kernel(c,a,b)
    kernel.launch(grid= (m*n//threads_per_block,1,1),block = (threads_per_block,1,1))

# @cute.kernel
# def vectorized_elementwise_kernel(c:cute.Tensor,a:cute.Tensor,b:cute.Tensor):
#     tidx,_,_ = cute.arch.thread_idx()
#     bidx,_,_ = cute.arch.block_idx()
#     dimx,_,_ = cute.arch.block_dim()
#     thread_idx = bidx*dimx+tidx

#     mi = thread_idx//n
#     ni = thread_idx%n
#     a_val = a[None,(mi,ni)].load()
#     b_val = b[None,(mi,ni)].load()
#     c[None,(mi,ni)] = a_val+b_val

# @cute.jit
# def vectorized_elementwise_add(c:cute.Tensor,a:cute.Tensor,b:cute.Tensor):
#     threads_per_block = 256
#     gA = cute.zipped_divide(a,tiler = (1,4))
#     gB = cute.zipped_divide(b,tiler = (1,4))
#     gC = cute.zipped_divide(c,tiler = (1,4))
#     print("[DSL INFO] Tiled Tensors:")
#     print(f"[DSL INFO]   gA = {gA}")
#     print(f"[DSL INFO]   gB = {gB}")
#     print(f"[DSL INFO]   gC = {gC}")
#     vectorized_elementwise_kernel(gC,gA,gB).launch(grid = (cute.size(gC,mode=[1])//thread_per_block,1,1),block = (threads_per_block,1,1),)

@cute.kernel
def vectorized_elementwise_add_kernel(
    gC: cute.Tensor,
    gA: cute.Tensor,
    gB: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx

    # Map thread index to logical index of input tensor in unit of vector
    m, n = gA.shape[1]  # thread-domain
    ni = thread_idx % n
    mi = thread_idx // n

    # Map logical index to physical address via tensor layout
    a_val = gA[(None, (mi, ni))].load()
    b_val = gB[(None, (mi, ni))].load()
    print(f"[DSL INFO] sliced gA = {gA[(None, (mi, ni))]}")
    print(f"[DSL INFO] sliced gB = {gB[(None, (mi, ni))]}")

    # Perform element-wise addition
    gC[(None, (mi, ni))] = a_val + b_val

@cute.jit
def vectorized_elementwise_add(mC: cute.Tensor, mA: cute.Tensor, mB: cute.Tensor):
    threads_per_block = 256

    gA = cute.zipped_divide(mA, (1, 4))
    gB = cute.zipped_divide(mB, (1, 4))
    gC = cute.zipped_divide(mC, (1, 4))

    print("[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   gA = {gA}")
    print(f"[DSL INFO]   gB = {gB}")
    print(f"[DSL INFO]   gC = {gC}")

    vectorized_elementwise_add_kernel(gC, gA, gB).launch(
        grid=(cute.size(gC, mode=[1]) // threads_per_block, 1, 1),
        block=(threads_per_block, 1, 1),
    )
