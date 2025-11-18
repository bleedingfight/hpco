import torch
from itertools import product
from .conftest import skip_if_triton_missing
from tilelang.autotuner import AutoTuner
from ..tilelang_ops import softmax_kernel
import tilelang
import pytest


@skip_if_triton_missing
class TestTilelangSoftmaxFunctions:
    def test_oneline_softmax(self):
        M = 8192
        N = 8192
        kernel = softmax_kernel(M, N)
        dtype = torch.float16
        X = torch.randn(M, N, dtype=dtype, device="cuda")
        Y = kernel(X)
        Y_ref = X.softmax(dim=1)

        torch.testing.assert_close(Y, Y_ref, rtol=1e-2, atol=1e-2)

        # t1 = do_bench(lambda: X.softmax(dim=1), warmup=25, rep=100)
        # t2 = do_bench(lambda: kernel(X), warmup=25, rep=100)
        # print(f"torch latency: {t1:.3f} ms")
        # print(f"TileLang latency: {t2:.3f} ms")
        # print(f"Speedup: {t1 / t2:.3f}x")
