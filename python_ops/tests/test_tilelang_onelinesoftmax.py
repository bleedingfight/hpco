import torch
from .conftest import skip_if_triton_missing
from ..tilelang_ops import softmax_kernel


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
