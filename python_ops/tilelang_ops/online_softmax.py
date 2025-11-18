import tilelang as tl
from tilelang.profiler import do_bench
import tilelang.language as T
import torch
from typing import Callable


@tl.jit(out_idx=[1])
def softmax_kernel(M, N, dtype: str = "float16") -> Callable:
    # 每段计算至少处理8K个元素
    BN = min(tl.next_power_of_2(N), 8192)
    # 一行需要处理多少段
    NN = tl.cdiv(N, BN)
    accum_dtype = "float"
    scale = 1.44269504

    @T.prim_func
    def main(X: T.Tensor([M, N], dtype), Y: T.Tensor([M, N], dtype)):
        with T.Kernel(M, threads=128) as (i_m):
            x = T.alloc_fragment([BN], dtype)
            y = T.alloc_fragment([BN], dtype)
            lse = T.alloc_fragment([1], accum_dtype)
            max_x = T.alloc_fragment([1], dtype)
            exp_x = T.alloc_fragment([BN], accum_dtype)
            sum_exp_x = T.alloc_fragment([1], accum_dtype)
            T.fill(lse, -T.infinity(accum_dtype))
            # 循环处理每一段数据(一次计算获取一行数据的最大值和指数求和分母)
            for i_n in T.Pipelined(0, NN):
                T.copy(X[i_m, i_n * BN : (i_n + 1) * BN], x)
                T.reduce_max(x, max_x, dim=0, clear=True)
                for j in T.Parallel(BN):
                    # 计算指数部分
                    exp_x[j] = T.exp2(x[j] * scale - max_x[0] * scale)
                # 计算这一段的指数求和
                T.reduce_sum(exp_x, sum_exp_x, dim=0, clear=True)
                # 更新分母
                lse[0] = max_x[0] * scale + T.log2(
                    T.exp2(lse[0] - max_x[0] * scale) + sum_exp_x[0]
                )
            for i_n in T.Pipelined(0, NN):
                # 复用寄存器
                T.copy(X[i_m, i_n * BN : (i_n + 1) * BN], x)
                for j in T.Parallel(BN):
                    y[j] = T.exp2(x[j] * scale - lse[0])
                T.copy(y, Y[i_m, i_n * BN : (i_n + 1) * BN])

    return main
