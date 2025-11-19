import tilelang as tl
import tilelang.language as T


@tl.jit(out_idx=[-1])
def native_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @T.prim_func
    def main(
        A: T.Tensor((K,), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor(
            N,
        ),
        dtype,
    ):
        with T.Kernel(T.ceildiv(K, BLOCK_K)) as bn:
            # threadIdx.x
            tn = T.get_thread_binding(0)
            A_shared = T.alloc_shared((BLOCK_K,), dtype)
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            C_reg = T.alloc_local((1,), accum_dtype)
            T.clear(C_reg)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                # 加载block_K的数据到共享内存
                for tk in T.serial(BLOCK_K):
                    A_shared[tk] = A[bk * BLOCK_K + tk]
                    B_shared[tn, tk] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
                # 计算block_K 和 block_Kxblock_N的乘加
                for tk in T.serial(BLOCK_K):
                    C_reg[0] += A_shared[tk].astype(accum_dtype) * B_shared[
                        tn, tk
                    ].astype(accum_dtype)
            C[bn * BLOCK_N + tn] = C_reg[0]
        return main
