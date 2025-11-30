import torch
from itertools import product
from ..cutedsl_ops import naive_elementwise_add,vectorized_elementwise_add
import pytest
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

class TestCuteFunctions:
    def test_cute_elementwise_add(self):
        m,n = 16386,2048
        a = torch.randn((m,n),dtype = torch.float16,device = "cuda")
        b = torch.randn((m,n),dtype = torch.float16,device = "cuda")
        c = torch.randn((m,n),dtype = torch.float16,device = "cuda")
        fn = cute.compile(naive_elementwise_add,from_dlpack(c,16),from_dlpack(a,16),from_dlpack(b,16))
        fn(c,a,b)
        torch.testing.assert_close(c,a+b)
    def test_cute_vectorized_elementwise_add(self):
        m,n = 16386,2048
        a = torch.randn((m,n),dtype = torch.float16,device = "cuda")
        b = torch.randn((m,n),dtype = torch.float16,device = "cuda")
        c = torch.randn((m,n),dtype = torch.float16,device = "cuda")
        fn = cute.compile(vectorized_elementwise_add,from_dlpack(c,16),from_dlpack(a,16),from_dlpack(b,16))
        fn(c,a,b)
        torch.testing.assert_close(c,a+b)
