import tilelang
import tilelang.language as T
import torch
from typing import Callable


@tilelang.jit(out_idx=-1)
def softmax_kernel(M, N, dtype: str = "float16") -> Callable:
    BN = min(T.next_power_of_2(N), 8192)
    NN = T.cdiv(N, BN)
    accum_dtype = "float"
    scale = 1.44269504

    @T.prim_func
    def main(X: T.Tensor([M, N], dtype), Y: T.Tensor([M, N], dtype)):
        with T.kernel(M, thread=128) as (i_m):
            x = T.alloc_fragment([BN], dtype)
            y = T.alloc_fragment([BN], dtype)
            lse = T.alloc_fragment([1], accum_dtype)
            max_x = T.alloc_fragment([1], dtype)
            exp_x = T.alloc_fragment([BN], accum_dtype)
            sum_exp_x = T.alloc_fragment([1], accum_dtype)
            T.fill(lse, -T.infinity(accum_dtype))
            for i_n in T.Pipeline(0, NN):
                T.copy(X[i_m, i_n * BN : (i_n + 1) * BN], x)
                T.reduce_max(x, max_x, dim=0, clear=True)
                for j in T.Parallel(BN):
                    exp_x[j] = T.exp2(x[j] * scale - max_x[0] * scale)
                T.reduce_sum(exp_x, sum_exp_x, dim=0, clear=True)
            for i_n in T.Pipelined(0, NN):
                T.copy(x[i_m, i_n * BN : (i_n + 1) * BN], x)
                for j in T.Parallel(BN):
                    y[j] = T.exp2(x[j] * scale - lse[0])
                T.copy(y, Y[i_m, i_n * BN : (i_n + 1) * BN])

    return main
