from cutlass.cute.runtime import from_dlpack
import numpy as np
import sys
import torch
import os
sys.path.append(os.path.abspath('..'))
from cutedsl_ops import naive_elementwise_add
import cutlass.cute as cute
from typing import Callable
def benchmark(callable, a_, b_, c_):
    avg_time_us = cute.testing.benchmark(
        callable,
        kernel_arguments=cute.testing.JitArguments(a_, b_, c_),
        warmup_iterations=5,
        iterations=100,
    )

    # Calculate metrics
    # ----------------
    dtype = a_.element_type
    num_elements = np.prod(a.shape)

    # Calculate total bytes transferred:
    # - 2 reads (A and B) + 1 write (C)
    # - Each element is dtype.width bits
    bytes_per_element = dtype.width // 8
    total_bytes = num_elements * bytes_per_element

    # Calculate achieved bandwidth
    achieved_bandwidth = total_bytes / (avg_time_us * 1000)  # GB/s

    # Print results
    # ------------
    print(f"Performance Metrics:")
    print(f"-------------------")
    print(f"Kernel execution time: {avg_time_us:.4f} us")
    print(f"Memory throughput: {achieved_bandwidth:.2f} GB/s")

if __name__ == "__main__":
    m,n = 16386,2048
    a = from_dlpack(torch.randn(m,n,dtype=torch.float16,device='cuda'),16)
    b = from_dlpack(torch.randn(m,n,dtype=torch.float16,device='cuda'),16)
    c = from_dlpack(torch.randn(m,n,dtype=torch.float16,device='cuda'),16)
    naive_elementwise_add = cute.compile(naive_elementwise_add,c,a,b)
    benchmark(naive_elementwise_add, c, a, b)

