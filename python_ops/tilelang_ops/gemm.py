import tilelang
import tilelang.language as T


def matmul(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    dtype="float16",
    accum_dtype="float",
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N)) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_local((block_M, block_N), accum_dtype)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                for k, j in T.Parallel(block_K, block_N):
                    B_shared[k, j] = B[ko * block_K + k, by * block_N + j]
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


if __name__ == "__main__":
    import torch

    func = matmul(1024, 1024, 1024, 128, 128, 128)
    jit_kernel = tilelang.compile(func, out_idx=[2], target="cuda")
    # 3. Prepare input tensors in PyTorch
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

    # 4. Invoke the JIT-compiled kernel
    c = jit_kernel(a, b)
    ref_c = a @ b

    # 5. Validate correctness
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel output matches PyTorch reference.")

    # 6. Inspect generated CUDA code (optional)
    cuda_source = jit_kernel.get_kernel_source()
    print("Generated CUDA kernel:\n", cuda_source)

    # 7. Profile performance
    profiler = jit_kernel.get_profiler()
    latency = profiler.do_bench()
    print(f"Latency: {latency} ms")
