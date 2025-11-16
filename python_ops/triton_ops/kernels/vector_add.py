import triton
import torch
import triton.language as tl


@triton.jit
def add_kernel(output_ptr, a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_offset = BLOCK_SIZE * pid
    offset = block_offset + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    a = tl.load(a_ptr + offset, mask=mask)
    b = tl.load(b_ptr + offset, mask=mask)
    c = tl.add(a, b)
    tl.store(output_ptr + offset, c)


def add(a, b):
    out = torch.empty_like(a)
    N = a.numel()
    grid = lambda meta: (triton.cdiv(a.numel(), meta["BLOCK_SIZE"]),)
    add_kernel[grid](out, a, b, N, BLOCK_SIZE=1024)
    return out


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2**i for i in range(12, 28, 1)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["triton", "torch"],  # Possible values for `line_arg`.
        line_names=["Triton", "Torch"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="vector-add-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device=torch.device("cuda"), dtype=torch.float32)
    y = torch.rand(size, device=torch.device("cuda"), dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: add(x, y), quantiles=quantiles
        )
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    a = torch.randn(10000).cuda()
    b = torch.randn(10000).cuda()
    c = add(a, b)
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=torch.device("cuda"))
    y = torch.rand(size, device=torch.device("cuda"))
    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    print(
        f"The maximum difference between torch and triton is "
        f"{torch.max(torch.abs(output_torch - output_triton))}"
    )
    benchmark.run(print_data=True, show_plots=True)
