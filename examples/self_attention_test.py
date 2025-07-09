import torch
from torch.nn import functional as F

batch_size, sequence_length, hidden_size = 1, 128, 64
query = torch.rand(
    batch_size, sequence_length, hidden_size, dtype=torch.float16, device="cuda"
)
key = torch.rand(
    batch_size, sequence_length, hidden_size, dtype=torch.float16, device="cuda"
)
value = torch.rand(
    batch_size, sequence_length, hidden_size, dtype=torch.float16, device="cuda"
)
out = F.scaled_dot_product_attention(query, key, value)
print(out)
# with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
#     F.scaled_dot_product_attention(query, key, value)
