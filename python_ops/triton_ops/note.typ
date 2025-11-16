= Softmax指数优化
$f(x_i) = frac(e^(x-overline(m)),sum_0^N e^(x_i-overline(m)))$
这里max(x)是一个全局结果：只有计算完一整行才能得到最终结果。对于每个block中有32个元素计算，200行的结果来说。需要循环7次，最后一次的时候元素个数为8。如果我们计算的时候使用每段的最大值作为实际值。我们可以这样假设
#table(
  columns: 6,
  [段需要], [当前最大值], [全局最大值], [差异],[分子真实值],[等价转换],
  [1], [$m_1$], [$overline(m)$], [$x_1-overline(m)$],[$x_1-overline(m)$],[$x_1-m_1+m_1-overline(m)$],
  [2], [$m_2$], [$overline(m)$], [$x_2-overline(m)$],[$x_2-overline(m)$],[$x_2-m_2+m_2-overline(m)$],
  [3], [$m_3$], [$overline(m)$], [$x_3-overline(m)$],[$x_3-overline(m)$],[$x_3-m_3+m_3-overline(m)$],
  [4], [$m_4$], [$overline(m)$], [$x_4-overline(m)$],[$x_4-overline(m)$],[$x_4-m_4+m_4-overline(m)$],
  [5], [$m_5$], [$overline(m)$], [$x_5-overline(m)$],[$x_5-overline(m)$],[$x_5-m_5+m_5-overline(m)$],
  [6], [$m_6$], [$overline(m)$], [$x_6-overline(m)$],[$x_6-overline(m)$],[$x_6-m_6+m_6-overline(m)$],
  [7], [$m_7$], [$overline(m)$], [$x_7-overline(m)$],[$x_7-overline(m)$],[$x_7-m_7+m_7-overline(m)$],
)
如果我们想计算一整行的结果（公式中的分子）本质上是计算[$x_1-overline(m)$,$x_2-overline(m)$,$x_3-overline(m)$,$x_4-overline(m)$,$x_5-overline(m)$,$x_6-overline(m)$,$x_7-overline(m)$]。于是对于第一段数据$x_1-overline(m) = x_1-m_1+m_1-overline(m)$，这样我们迭代7轮可以得到最后的全局最大值$overline(m)$，于是我们求$f(x_i) = frac(e^(x-overline(m)),sum_0^N e^(x_i-overline(m)))$。分子计算可以转换为：
$e^(x_1-overline(m)) = e^(x_1-m_1+m_1-overline(m)) = e^(x_1-m_1)e^(m_1-overline(m))$。于是我们对指数求和的运算: $ sum(e^(x_1-overline(m))) &= e^(x_1-overline(m))+ e^(x_2-overline(m))+e^(x_3-overline(m))+e^(x_4-overline(m))+e^(x_5-overline(m))+e^(x_6-overline(m))+e^(x_7-overline(m)) \ & = e^(x_1-m_1+m_1-overline(m))+e^(x_2-m_2+m_2-overline(m))+e^(x_3-m_3+m_3-overline(m))\ 
&+e^(x_4-m_4+m_4-overline(m))+e^(x_5-m_5+m_5-overline(m))+e^(x_6-m_6+m_6-overline(m))+e^(x_7-m_7+m_7-overline(m)) $ 

我们定义：$f_i = e^(m_i-overline(m))$，整个求和公式变成了：
$  sum(e^(x_1-overline(m))) &= f_1e^(x_1-m_1)+f_2e^(x_2-m_2)+f_3e^(x_3-m_3)+f_4e^(x_4-m_4)+ \ &f_5e^(x_5-m_5)+f_6e^(x_6-m_6)+ f_7e^(x_7-m_7) $

举个例子：现在有两段数据，第一段数据计算出的最大值是$m_1$，第二段计算出的结果是$m_2$，$m_2>m_1$，则全局最大值$overline(m) = m_2$。我们的真实结果应该是$ f(x) = e^(x_1-m_2)+e^(x_2-m_2) $

现在我们可以先按照局部最大值算出第一段数据的结果：$f(x) = e^(x_1-m_1)$。计算第二段的时候因为$m_2=overline(m)$，所以第二段的结果是正确的。此时我们计算出的结果为：$ f^*(x)  = e^(x_1-m_1)+e^(x_w-m_2) $
我们对真实公式做一个简单地变形：$ f(x) = e^(x_1-m_2)+e^(x_2-m_2) = e^(x_1-m_1+m_1-m_2)+e^(x_2-m_2) = f*e^(x_1-m_1)+e^(x_2-m_2) $ 这里我们的$f = e^(m_1-m_2) = e^(m_1-overline(m))$ 其中$ f e^(x_1-m_1) $ 可以被理解为局部最大值小于全局最大值，所以使用局部最大值计算导致计算整体偏小，将差异以补偿因子的形式作用在这个局部最大值的结果之上就可以得到全局最大。
== 总结
指数优化的核心是每次需要减去全局最大值，但是每次计算的时候只能计算出当前段数据的最大值。在计算到最后一段数据之前都没法确定全局最大值。指数优化的核心是：每次使用当前最大值计算，当全局最大值更新之后对计算出的结果补偿差异，这样就可以一轮计算出全局最大值和指数求和的结果。

== FlashAttention的指数优化
Attention的计算公式:$ f(Q,K,V) = S o f t m a x(Q K^T)V $
这里计算的Q是元素为m的一维列向量，$K^T$是一维行向量。则$Q K^T$是$m times n$的矩阵。然后 V 也是包含n个元素的行向量。现在计算难点主要是$S o f t m a x(Q K^T)$这个矩阵需要使用和上面Softmax相同的做法运算。即对$m times n $的矩阵：
1. 每一行减去此行的最大值
2. 上一行的结果逐元素求指数
3. 处以指数求和的结果
最大值和求和都是reduce操作：#emph[没有遍历完所有的数据都无法得到全局结果]。
=== 多阶段SelfAttention
多阶段的SelfAttention计算如下：
矩阵Q的第k行$Q[k,:]$和矩阵$K^T$的第i列点乘运算得到输出O的$O[k][j]$的结果。这样我们遍历i从[0,N-1)(假设Q有N列)即可得到输出的第k行结果。这样我们可以：
1. 对于$i in[0,N)$，读取Q的一行数据分别和$K^T$的第i列运算，即可得出输出矩阵的第k列的各个结果(计算矩阵乘法同时计算全局最大值)。
 1. 读取Q第k列，计算向量的点积：$Q[k,:] \cdot K^T[:,i]$
 2. 将计算的结果和初始化全局最大值$m_0$比较：$m_i = max(m_{i-1},x_i)$
 3. 更新分母的值$d_i' = d_(i-1)'e^{m_{i-1}-m_i}+e^{x_i-m_i}$
2. 经过上一阶段的计算之后，全局最大值$m_N$已经计算得到，同时计算了一行的结果同样softmax的分母也得到了$d^'_N$：
 1. 计算softmax的分子/分母的结果：$frac(x_i-m_N,1)$其中$m_N$表示N个元素的全局最大值。
 2. 对这一行
最关键的是$o_i$的运算，假设：$ A_(m times n) = matrix(Q \cdot K^T) $
