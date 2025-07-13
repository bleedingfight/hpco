import torch
from torch.nn import functional as F
import math

torch.manual_seed(0)
batch_size, sequence_length, hidden_size = 2, 128, 64
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


def selfattention(Q, K, V):
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value


s_out = selfattention(query, key, value)

print(out, s_out)
# with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
#     F.scaled_dot_product_attention(query, key, value)
