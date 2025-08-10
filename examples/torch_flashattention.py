import torch
import torch.nn.functional as F
# Optionally use the context manager to ensure one of the fused kernels is run
query = torch.rand(8, 128, dtype=torch.float16, device="cuda")
key = torch.rand(8, 128, dtype=torch.float16, device="cuda")
value = torch.rand(8, 128, dtype=torch.float16, device="cuda")
output = F.scaled_dot_product_attention(query, key, value)
print(output.shape)
