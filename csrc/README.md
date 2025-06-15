# 使用纯CUDA实现的算子
## 实现算子elu 
f(x) = \begin{cases}
x & \text{if } x > 0 \\
\alpha(e^x - 1) & \text{if } x \le 0
\end{cases}
