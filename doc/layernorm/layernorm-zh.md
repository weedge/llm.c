# 层归一化

快速教程。让我们看看 LayerNorm 是如何处理的，作为模型中的一个示例层。我们从 [PyTorch LayerNorm 文档](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) 开始。LayerNorm 当然来自这篇原始论文 [Ba et al. 2016](https://arxiv.org/abs/1607.06450)，并且在 [Vaswani 等人](https://arxiv.org/abs/1706.03762) 著名的论文《Attention is All You Need》中被引入到 Transformer 中。[GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 采用了与 Transformer 相同的架构，但 LayerNorm 的位置被移动到了现在被称为预归一化版本的位置。也就是说，Transformer 的残差路径保持清晰，而 LayerNorm 现在是每个 Transformer 块的第一层。这显著提高了训练的稳定性。

当查看 [PyTorch LayerNorm 文档](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) 时，首先要注意的是您很可能找不到实际实现的方程式。这是因为它被深埋在代码的 30 层深处，位于一个晦涩的动态分发器后面，在一些可能是自动生成的 CUDA 代码中（对于对细节感兴趣的人，请参见 [layer_norm.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/layer_norm.cpp) 和 [layer_norm_kernel.cu](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/layer_norm_kernel.cu)）。这是因为 PyTorch 确实非常非常关心效率，这是公平的。但是对于我们的目的来说，我们必须首先使用更简单的 PyTorch 操作手动实现 LayerNorm。这将比仅转发 `LayerNorm` 模块要低效得多，但在算法上是有教益的。所以这是使用更简单的 PyTorch 操作直接实现 LayerNorm 数学的方法：

```python
import torch
eps = 1e-5

class LayerNorm:

    @staticmethod
    def forward(x, w, b):
        # x 是输入激活，形状为 B,T,C
        # w 是权重，形状为 C
        # b 是偏置，形状为 C
        B, T, C = x.size()
        # 计算均值
        mean = x.sum(-1, keepdim=True) / C # B,T,1
        # 计算方差
        xshift = x - mean # B,T,C
        var = (xshift**2).sum(-1, keepdim=True) / C # B,T,1
        # 计算标准差的倒数：**0.5 是 sqrt，**-0.5 是 1/sqrt
        rstd = (var + eps) ** -0.5 # B,T,1
        # 对输入激活进行归一化
        norm = xshift * rstd # B,T,C
        # 在最后对归一化的激活进行缩放和偏移
        out = norm * w + b # B,T,C

        # 返回输出和缓存，后者是后向传递期间稍后需要的变量
        cache = (x, w, mean, rstd)
        return out, cache
```

在训练期间，Transformer 的残差路径中的激活张量是 3 维数组（张量），形状为 `B,T,C`。其中 B 是批处理大小，T 是时间，C 是通道。例如，B=8，T=1024，C=768 是您可能看到的最小（124 百万参数）的 GPT-2 模型设置之一。

我们可以使用一些随机数进行前向传递：

```python
B = 2 # 这里使用一些玩具数字
T = 3
C = 4
x = torch.randn(B, T, C, requires_grad=True)
w = torch.randn(C, requires_grad=True)
b = torch.randn(C, requires_grad=True)
out, cache = LayerNorm.forward(x, w, b)
```

我们得到的是张量 `out`，也是形状为 `B,T,C` 的张量，其中每个 C 维“纤维”激活被归一化，然后缩放，并在最后也通过该层的权重和偏置偏移。重要的是要注意，我们还返回一个名为 `cache` 的变量，它是输入激活 `x`、权重 `w`、均值 `mean` 和倒数标准差 `rstd` 的元组。这些都是我们在后向传递期间需要的变量。

当然，PyTorch 可以为我们自动完成此层的后向传递，利用其 Autograd。让我们先来做这个：

```python
dout = torch.randn(B, T, C)
fakeloss = (out * dout).sum()
fakeloss.backward()
```

在这里，我们创建了一个 `fakeloss`，它只是将我们的 layernorm 的所有输出加权组合（随机）成一个标量值（损失），以便我们得到计算图的单个输出。通常，这将是模型的损失，但这里我们只是做了一个假的损失。然后我们对这个标量调用 `backward()`，PyTorch 将为我们计算出所有输入到此图的梯度 - 即输入激活 `x`、权重 `w` 和偏置 `b`。如果您对 autograd 不太了解，我建议您观看我的 [micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) 视频，我们在其中构建了一个微小的 autograd 引擎。因此，PyTorch autograd 的魔法是，在我们调用 `.backward` 后，它将使用该张量的 `.grad` 属性填充所有具有 `requires_grad=True` 的张量的梯度，这些梯度告诉我们损失对于该张量的所有输入数的斜率。因此，`x.grad`、`w.grad` 和 `b.grad` 的形状与 `x`、`w` 和 `b` 的形状完全相同。

但我们不想使用 PyTorch Autograd。我们想要手动进行后向传递。所以我们拿出笔和纸，写下了 LayerNorm 的表达式。前向传递具有以下数学形式：

$\text{LayerNorm}(x) = w \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + b$

其中 $\odot$ 是逐元素乘法，$\mu$ 是均值，$\sigma^2$ 是方差，$\epsilon$ 是一个小常数，以避免除以零。记住微积分中的导数规则，现在我们想要导出梯度。对于这部分，我的视频 [Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) 可能会非常有帮助，因为我详细地解释了一个类似的层 - 批归一化层。当你计算梯度时，你会注意到表达式在解析上简化了，你可以移动项并简化表达式。所以你不必手动反向传递前向传递中的每一行。特别地，我们得到：

```python
    @staticmethod
    def backward(dout, cache):
        x, w, mean, rstd = cache
        # 重新计算 norm（节省内存但需要计算）
        norm = (x - mean) * rstd
        # 权重、偏置的梯度
        db = dout.sum((0, 1))
        dw = (dout * norm).sum((0, 1))
        # 输入的梯度
        dnorm = dout * w
        dx = dnorm - dnorm.mean(-1, keepdim=True) - norm * (dnorm * norm).mean(-1, keepdim=True)
        dx *= rstd
        return dx, dw, db
```

因此，给定存储在 `dout` 中的每个单独输出数上的梯度，以及来自前向传递的 `cache`，我们现在可以通过这个层向输入进行反向传递，以继续后向传递的链式法则。现在我们可以进行自己的后向传递，并看到它们匹配（误差非常小）：

```python
dx, dw, db = LayerNorm.backward(dout, cache)
print("dx 误差:", (x.grad - dx).abs().max().item())
print("dw 误差:", (w.grad - dw).abs().max().item())
print("db 误差:", (b.grad - db).abs().max().item())
```

注意一件事。在后向传递中，我们重新计算了变量 `norm`。我们在前向传递中已经计算了这个变量，但是然后我们扔掉了它！我们不能将其作为一部分保存到 `cache` 中吗，以避免这种重新计算？事实上，我们完全可以，并且当然会得到完全相同的结果。我们保存到 `cache` 中的内容量完全取决于我们。我们甚至不必保存 `mean` 和 `rstd`，我们可以在后向传递中重新计算它们。区别在于 `mean` 和 `rstd` 非常小，仅形状为 `B,T`，而 `norm` 的形状为 `B,T,C`。因此，这只是内存和计算之间的权衡。通过不将 `norm` 保留在缓存中，我们节省了内存，但是我们用稍后的后向传递换取了一些计算量。这在所有层中都非常常见，您将看到深度学习框架中各种层的不同实现可能具有不同的“检查点设置”。是的，令人困惑的是，这被称为检查点，并且与将模型权重保存到磁盘无关。这是关于在前向传递中保存中间变量以在后向传递中节省计算量的。

好了，这就是 PyTorch 张量版本。现在我们必须将其转移到 C 中，并摆脱张量抽象。在我给出完整的前向传递实现之前，先简要介绍一下张量。张量是什么？它们是 1) 一个称为 Storage 的 1D 内存块，其中包含原始数据，和 2) 对该存储的形状的视图。[PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/) 在这里可能会有所帮助。所以例如，如果我们有 3D 张量：

```python
torch.manual_seed(42)
B, T, C = 2, 3, 4
a = torch.randn(B, T, C)
print(a)

tensor([[[ 1.9269,  1.4873,  0.9007, -2.1055],
         [ 0.6784, -1.2345, -0.0431, -1.6047],
         [ 0.3559, -0.6866, -0.4934,  0.2415]],

        [[-1.1109,  0.0915, -2.3169, -0.2168],
         [-0.3097, -0.3957,  0.8034, -0.6216],
         [-0.5920, -0.0631, -0.8286,  0.3309]]])
```

这是一个 2x3x4 张量，但它的底层内存只有一个尺寸为 2\*3\*4=24 的 1D 数组。View 只是这个 1D 数组上的形状。所以现在当我们索引到这个 PyTorch 张量时，例如 `a[1,2,3]`，PyTorch 将计算到 1D 数组的偏移量为 `1*3*4 + 2*4 + 3 = 23`，并返回该偏移量处的值。一般的公式是，如果你想要检索任何元素 `b,t,c`，你将其偏移为 `b*T*C + t*C + c`。所以例如：

```python
b,t,c = 1,2,3
print(a[b,t,c])
print(a.view(-1)[b*T*C + t*C + c])
```

这两者都打印出了 0.3309。所以通过这种方式，我们知道如何访问所有单独的元素，以及如何偏移所有指针。特别地，注意通道维度是最内部的维度。所以当我们增加偏移量 1 时，我们在遍历通道维度。这对于我们 C 实现的内存布局是很重要的考虑。前向传递的等效 C 代码如下：

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // 指向输入位置 inp[b,t,:] 
            float* x = inp + b * T * C + t * C;
            // 计算均值
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // 计算方差（没有任何偏差校正）
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // 计算 rstd
            float s = 1.0f / sqrtf(v + eps);
            // 指向输出位置 out[b,t,:] 
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // 归一化输出
                float o = n * weight[i] + bias[i]; // 缩放并偏移
                out_bt[i] = o; // 写入
            }
            // 缓存均值和 rstd 供后向传递使用
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}
```

您将看到我如何偏移到 `inp[b,t]`，然后您知道接下来的 `C` 元素是该位置（批次、时间）的通道。和后向传递：

```c
void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        float* dout, float* inp, float* weight, float* mean, float* rstd,
                        int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            float mean_bt = mean[b * T + t];
            float rstd_bt = rstd[b * T + t];

            // 首先：两个 reduce 操作
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * r

std_bt;
                dnorm_mean += dout_bt[i];
                dnorm_norm_mean += dout_bt[i] * norm_bti;
            }
            dnorm_mean *= (-rstd_bt);
            dnorm_norm_mean *= (-rstd_bt);

            // 偏置和权重的梯度
            for (int i = 0; i < C; i++) {
                dbias[i] += dout_bt[i];
                dweight[i] += dout_bt[i] * ((inp_bt[i] - mean_bt) * rstd_bt);
            }

            // 计算 dnorm
            float dnorm[C];
            for (int i = 0; i < C; i++) {
                dnorm[i] = dout_bt[i] * weight[i];
            }
            // 计算 dinp
            for (int i = 0; i < C; i++) {
                dinp_bt[i] = dnorm[i] * rstd_bt + dnorm_mean / C + inp_bt[i] * dnorm_norm_mean / C;
            }
        }
    }
}
```

您将看到此处的内存布局非常重要。对于 `layernorm_backward`，我们需要在给定 batch、time 和 channel 的情况下访问 `mean` 和 `rstd`。我们在这里访问的元素具有形状 `(B, T)`。因此，我们通过直接索引 `mean[b * T + t]` 来获得正确的均值和标准差。同样，对于 `dout`、`inp` 等，我们必须使用相同的索引技巧来访问正确的张量元素。

这是一个 LayerNorm 的基本实现。在实践中，您可能需要添加更多的功能，如输入掩码（masking）、多头注意力等。但这应该给您一个很好的起点。希望这可以帮助您理解如何实现这些层，以及在深度学习库之外，它们是如何构建的。