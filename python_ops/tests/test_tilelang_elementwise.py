import torch
from itertools import product
from .conftest import skip_if_triton_missing
from tilelang.autotuner import AutoTuner
from ..tilelang_ops import vec_add
import tilelang
import pytest


@skip_if_triton_missing
class TestTilelangFunctions:
    def test_tilelang_vec_add(self):
        M = 128
        N = 128
        a = torch.randn(M, N, dtype=torch.float32, device="cuda").cuda()
        b = torch.randn(M, N, dtype=torch.float32, device="cuda").cuda()
        out = vec_add(M, N, 32, 32, "float32", "float32", 256)(a, b)

        truth = a + b
        torch.testing.assert_close(out, truth, rtol=1e-2, atol=1e-2)

    def test_autotune(self):
        def get_configs(M, N):
            block_M = [64, 128, 256]
            block_N = [64, 128, 256]
            threads = [64, 128, 256]
            configs = list(product(block_M, block_N, threads))
            return [
                {"block_M": bm, "block_N": bn, "threads": th} for bm, bn, th in configs
            ]

        def get_best_config(M, N):
            def kernel(block_M=None, block_N=None, threads=None):
                return vec_add(M, N, block_M, block_N, "float32", "float32", threads)

            autotuner = (
                AutoTuner.from_kernel(kernel=kernel, configs=get_configs(M, N))
                .set_compile_args(
                    out_idx=[-1],
                    target="cuda",
                )
                .set_profile_args(
                    supply_type=tilelang.TensorSupplyType.Auto,
                    ref_prog=lambda x, y: x + y,
                    skip_check=False,
                )
            )
            return autotuner.run(warmup=3, rep=20)

        # for m, n in [(128, 128), (256, 256), (512, 512)]:
        for m, n in [(128, 128)]:
            result = get_best_config(m, n)
            kernel = result.kernel
            a = torch.randn(m, n, dtype=torch.float32, device="cuda")
            b = torch.randn(m, n, dtype=torch.float32, device="cuda")
            out = kernel(a, b)
            torch.testing.assert_close(out, a + b, rtol=1e-2, atol=1e-2)
