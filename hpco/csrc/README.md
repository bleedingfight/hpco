# 使用纯CUDA实现的算子

## 实现算子elu

f(x) = \begin{cases}
x & \text{if } x > 0 \\
\alpha(e^x - 1) & \text{if } x \le 0
\end{cases}

## 实现算子embedding

embedding本质上相当于给定数据，使用输入在对应的权重维度上选择数据。输出数据的layout和输入layout相同。PyTorch中函数签名如下:

```python
torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None)

```

其中:

-. `num_embeddings`:embedding矩阵的行数。
-. `embedding_dim`:embedding矩阵的列数。

- `padding_idx`:填充那一行为0。
- `max_norm`:如果设置了则选中的行的norm如果大于max_norm则需要对结果norm保证其norm的结果不能大于max_norm。
- `norm_type`:norm类型，通常是2范数。
- `scale_grad_by_freq`：
- `sparse`:如果为True则embedding的weight将按照梯度存储。

## Silu

$SiLU(x) = x\cdot\frac{1}{1+e^{-x}}$

## 实现算子LayerNorm

LayerNorm对高维数据的计算：

1. 沿着最后一个维度求出均值和方差
2.
