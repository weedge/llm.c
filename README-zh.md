# llm.c

LLM在简单的纯C/CUDA中进行训练。没有必要使用245MB的PyTorch或107MB的cPython。例如，训练GPT-2（CPU，fp32）需要约1000行清晰的代码，放在单个文件中。它可以立即编译和运行，并且与PyTorch参考实现完全匹配。我选择GPT-2作为第一个工作示例，因为它是LLM的祖师爷，是现代技术堆栈首次组合在一起的标志。

目前，我的工作重点在于：

- 直接CUDA实现，这将显着提高速度，可能接近PyTorch。
- 使用SIMD指令加速CPU版本，x86上的AVX2 / ARM（例如苹果硅）上的NEON。
- 更现代的架构，例如Llama2、Gemma等。

对于存储库，我希望在保持简洁、简单的参考实现的同时，还维护着更多优化版本，这些版本可以接近PyTorch的性能，但代码和依赖项却只是一小部分。

## 快速开始

下载并标记一个数据集。[tinyshakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) 数据集是下载和标记速度最快的：

```bash
python prepro_tinyshakespeare.py
```

这将打印：

```
Saved 32768 tokens to data/tiny_shakespeare_val.bin
Saved 305260 tokens to data/tiny_shakespeare_train.bin
```

.bin文件是GPT-2分词器表示的token id的原始字节流。或者，您也可以使用`prepro_tinystories.py`标记[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)数据集。

原则上，我们已经准备好从这里开始训练模型了。但是基线CPU/fp32参考代码效率低下，因此尚不实际从头开始训练这些模型。相反，我们使用由OpenAI发布的GPT-2权重进行初始化，并进行微调。为此，我们必须下载GPT-2权重并将它们保存为可以在C中加载的检查点：

```bash
python train_gpt2.py
```

您将会认出这段代码来自nanoGPT，它是一个简单的PyTorch中的GPT-2参考实现。此脚本将下载GPT-2（124M）模型，对一批数据进行10次迭代的过拟合，运行几步生成，最重要的是它将保存两个文件：1）包含在C中加载的原始模型权重的`gpt2_124M.bin`文件，以及`gpt2_124M_debug_state.bin`，它还包含更多的调试状态：输入、目标、logits和损失。这对于调试C代码、单元测试和确保我们与PyTorch参考实现完全匹配非常有用。现在我们只关心`gpt2_124M.bin`中的模型权重。我们现在可以用它们初始化并在原始的C中训练。首先编译代码：

```bash
make train_gpt2
```

您可以查看`Makefile`及其注释。它将尝试自动检测您的系统上是否有OpenMP，这对于以非常低的代码复杂性加速代码非常有帮助。一旦编译了`train_gpt2`，您可以运行它：

```bash
OMP_NUM_THREADS=8 ./train_gpt2
```

您应该根据您的CPU有多少核心来调整线程数量。程序将加载模型权重、tokens，它将使用Adam lr 1e-4运行几次微调循环，然后从模型中生成一个样本。文件（我认为）非常易读，您应该看一下。简而言之，所有层的前向和反向传递都有实现，并且它们被串联在一个大的、手动的前向/反向/更新循环中。在我的MacBook Pro（苹果硅M3 Max）上，输出如下所示：

```
[GPT-2]
max_seq_len: 1024
vocab_size: 50257
num_layers: 12
num_heads: 12
channels: 768
num_parameters: 124439808
train dataset num_batches: 1192
val dataset num_batches: 128
num_activations: 73323776
val loss 5.252026
step 0: train loss 5.356189 (took 1452.121000 ms)
step 1: train loss 4.301069 (took 1288.673000 ms)
step 2: train loss 4.623322 (took 1369.394000 ms)
step 3: train loss 4.600470 (took 1290.761000 ms)
...（已截断）...
step 39: train loss 3.970751 (took 1323.779000 ms)
val loss 4.107781
generated: 50256 16773 18162 21986 11 198 13681 263 23875 198 3152 262 11773 2910 198 1169 6002 6386 2583 286 262 11858 198 20424 428 3135 7596 995 3675 13 198 40 481 407 736 17903 11 329 703 6029 706 4082 198 42826 1028 1128 633 263 11 198 10594 407 198 2704 454 680 1028 262 1027 28860 286 198 3237 323
step 40: train loss 4.377757 (took 1366.368000 ms)
```

生成现在只给出token id，我们需要将其解码回文本。我们也可以在C中很容易地实现这一点，因为解码非常简单，只是字符串块查找和打印。现在我们可以使用tiktoken：

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
print(enc.decode(list(map(int, "50256 16773 18162 21986 11 198 13681 263 23875 198 3152 262 11773 2910 198 1169 6002 6386 2583 286 262 11858 198 20424 428 3135 7596 995 3675 13 198 40 481 407 736 17903 11 329 703 6029 706 4082 198 42826 1028 1128 633 263 11 198 10594 407 198 2704 454 680 1028 262 1027 28860 286 198 3237 323".split()))))
```

这将打印：

```
Come Running Away,
Greater conquer
With the Imperial blood
the heaviest host of the gods
into this wondrous world beyond.
I will not back thee, for how sweet after birth
Netflix against repounder,
will not
flourish against the earlocks of
Allay
```

我喜欢Netflix这个词，显然训练的阴影仍然潜伏在模型中。我没有尝试调整微调超参数，所以很可能这还可以大大改善，特别是如果要训练更长一些时间的话。

## 测试

我还附上了一个简单的单元测试，以确保我们的C代码与PyTorch代码一致。编译并运行：

```
make test_gpt2
./test_gpt2
```

现在这将加载`gpt2_124M_debug_state.bin`文件，运行前向传递，将logits和损失与PyTorch参考实现进行比较，然后进行10次Adam训练迭代，并确保损失与PyTorch匹配。

## 教程

我在这里附上了一个非常小的教程，在[doc/layernorm/layernorm.md](doc/layernorm/layernorm-zh.md)中。这是一个关于如何实现GPT-2模型的单个层，即layernorm层的简单、逐步指南。这是理解C中层如何实现的一个很好的起点。

## 许可

MIT