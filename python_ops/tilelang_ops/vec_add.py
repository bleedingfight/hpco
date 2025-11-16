import torch
import tilelang.language as T
import tilelang


@tilelang.jit(out_idx=[-1])
def vec_add(M, N, block_M, block_N, in_dtype, out_dtype, threads):
    @T.prim_func
    def elem_add(
        A: T.Tensor((M, N), dtype=in_dtype),
        B: T.Tensor((M, N), dtype=in_dtype),
        C: T.Tensor((M, N), dtype=out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            a_shared = T.alloc_shared((block_M, block_N), dtype=in_dtype)
            b_shared = T.alloc_shared((block_M, block_N), dtype=in_dtype)
            c_local = T.alloc_fragment((block_M, block_N), dtype=out_dtype)
            c_shared = T.alloc_shared((block_M, block_N), dtype=out_dtype)
            T.copy(A[by * block_M, bx * block_N], a_shared)
            T.copy(B[by * block_M, bx * block_N], b_shared)
            for local_y, local_x in T.Parallel(block_M, block_N):
                c_local[local_y, local_x] = (
                    a_shared[local_y, local_x] + b_shared[local_y, local_x]
                )
            T.copy(c_local, c_shared)
            T.copy(c_shared, C[by * block_M, bx * block_N])

    return elem_add
