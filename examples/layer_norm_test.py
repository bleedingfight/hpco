import torch
from torch import nn
# def layer_norm((x,norm_shape=None,eps=1e-5)):
#     if norm_shape is None:
#         norm_shape = x.shape[-1:]
#     return nn.LayerNorm(norm_shape, eps=eps)(x)


def layer_norm_custom(input, normalized_shape, bias, weight, eps=1e-5):
    norm_size = normalized_shape.product()
    mean = input.mean(dim=-1, keepdim=True)
    std = input.std(dim=-1, keepdim=True)
    return weight * (input - mean) / (std + eps) + bias


batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)
layer_norm = nn.LayerNorm(embedding_dim)
breakpoint()
# Activate module
out = layer_norm(embedding)
print(out)
