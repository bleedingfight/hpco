import torch
from torch.utils._triton import has_triton
from torch.library import triton_op, wrap_triton

if not has_triton():
    print("Skipping because triton is not supported on this device.")
else:
    import triton
    from triton import language as tl

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 4}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE": 4}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_SIZE": 2}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE": 2}, num_stages=4, num_warps=4),
        ],
        key=[],
    )
    @triton.jit
    def elu_kernel(
        in_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr + offsets, mask=mask)
        alpha = 1.0
        output = tl.where(x >= 0, x, alpha * (tl.exp(x) - 1))
        tl.store(out_ptr + offsets, output, mask=mask)

    @torch.compile(fullgraph=True)
    def elu(x):
        output = torch.zeros_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        elu_kernel[grid](x, output, n_elements)
        return output

    @triton_op("triton_hpco::elu", mutates_args={})
    def elu_triton(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n_elements = out.numel()
        wrap_triton(elu_kernel)[(n_elements,)](x, out, n_elements)
        return out

    # x = torch.randn(100, device="cuda")
    # triton_out = elu(x)
    # truth = torch.nn.ELU()(x)

    x = torch.randn(100, device="cuda")
    triton_out = torch.ops.triton_hpco.elu.default(x)
    truth = torch.nn.ELU()(x)
    torch.testing.assert_close(triton_out, truth, atol=1e-6, rtol=1e-6)
    print(f"torch output = {truth} triton output = {triton_out}")
