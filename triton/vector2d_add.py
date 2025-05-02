import triton
import triton.language as tl
import torch


@triton.jit
def add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,  # Pointers to input and output tensors
    n_cols: tl.constexpr,  # Number of columns (static for the kernel)
    n_rows: tl.constexpr,  # Number of rows (static for the kernel)
    BLOCK_SIZE_COL: tl.constexpr,  # Tile size for columns
    BLOCK_SIZE_ROW: tl.constexpr,  # Tile size for rows
):
    """Kernel for element-wise addition of two 2D matrices."""

    # Calculate the starting row and column index for this program instance
    # 获取计算矩阵的行和列索引
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    # Calculate the block start indices
    row_start = pid_row * BLOCK_SIZE_ROW
    col_start = pid_col * BLOCK_SIZE_COL

    # Create index ranges for the current block
    rows = row_start + tl.arange(0, BLOCK_SIZE_ROW)
    cols = col_start + tl.arange(0, BLOCK_SIZE_COL)

    # Create masks to handle boundary conditions (when block goes beyond matrix dims)
    row_mask = rows < n_rows
    col_mask = cols < n_cols
    mask = row_mask[:, None] & col_mask[None, :]  # Combine masks for 2D

    # Pointers to the current block in the input matrices
    a_block_ptr = a_ptr + rows[:, None] * n_cols + cols[None, :]
    b_block_ptr = b_ptr + rows[:, None] * n_cols + cols[None, :]
    c_block_ptr = c_ptr + rows[:, None] * n_cols + cols[None, :]

    # Load the current block from the input matrices
    a_block = tl.load(a_block_ptr, mask=mask, other=0.0)
    b_block = tl.load(b_block_ptr, mask=mask, other=0.0)

    # Perform the element-wise addition
    c_block = a_block + b_block

    # Store the result back to the output matrix
    tl.store(c_block_ptr, c_block, mask=mask)


def add_matrices(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Wrapper function to launch the Triton kernel for matrix addition."""
    assert a.shape == b.shape, "Input matrices must have the same shape."
    n_rows, n_cols = a.shape
    c = torch.empty_like(a)

    # Determine the block sizes for the kernel (tunable for performance)
    BLOCK_SIZE_COL = 256
    BLOCK_SIZE_ROW = 64

    # Calculate the grid size for launching the kernel
    grid_size_cols = (n_cols + BLOCK_SIZE_COL - 1) // BLOCK_SIZE_COL
    grid_size_rows = (n_rows + BLOCK_SIZE_ROW - 1) // BLOCK_SIZE_ROW
    grid = (grid_size_rows, grid_size_cols)

    # Launch the Triton kernel
    add_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        n_cols=n_cols,
        n_rows=n_rows,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
    )
    return c


if __name__ == "__main__":
    # Example usage
    shape = (512, 1024)
    a = torch.randn(shape, device="cuda")
    b = torch.randn(shape, device="cuda")
    c_triton = add_matrices(a, b)
    c_torch = a + b

    # Verify the result
    torch.testing.assert_close(c_triton, c_torch, rtol=1e-05, atol=1e-08)
    print("Triton and Torch results match!")
