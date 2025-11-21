import torch
from itertools import product
from .conftest import skip_if_triton_missing
from tilelang.autotuner import AutoTuner
from ..tilelang_ops import naive_gemv
import tilelang
import pytest


@skip_if_triton_missing
class TestTilelangMV:
    def test_mv(self):
        N = 1024
        K = 1024
        A = torch.randn((K,), dtype=torch.float16).cuda()
        B = torch.randn((N, K), dtype=torch.float16).cuda()
        C = naive_gemv(N, K, 128, 128)(A, B)
        torch.testing.assert_close(C, A @ B.T, rtol=1e-2, atol=1e-2)
