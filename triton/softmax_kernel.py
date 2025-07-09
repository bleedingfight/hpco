import torch
from torch.utils._triton import has_triton
from torch.library import triton_op, wrap_triton

import triton
from triton import language as tl
from triton.runtime import driver


@triton.jit
def softmax_native_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + input_row_stride * row_idx
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offset
        mask = col_offset < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row_minus_max = row - tl.max(row, axis=0)
        num = tl.exp(row_minus_max)
        den = tl.sum(row_minus_max, axis=0)
        softmax_output = num / dev
        output_row_start_ptr = output_ptr + row_idx + output_row_stride
        output_ptrs = output_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


properties = driver.active.utils.get_device_properties(torch.device("cuda"))
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    y = torch.empty_like(x)
    kernel = softmax_native_kernel.warmup(
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_stages=num_stages,
        num_warps=num_warps,
        grid=(1,),
    )
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    num_programs = NUM_SM * occupancy
    num_programs = min(num_programs, n_rows)
    kernel[(num_programs, 1, 1)](
        y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages
    )
    return y


def native_softmax(x):
    x_max, _ = x.max(axis=1)
    z = x - x_max[:, None]
    num = torch.exp(z)
    den = torch.sum(num, axis=1)
    return num / den[:, None]


a = torch.randn(3, 4)
b = native_softmax(a)
b = softmax(a)
print(b)
