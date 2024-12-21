# 计算语言学大作业：探寻知识边界

## 项目环境配置

本小组的知识边界项目（Knowledge Boundary, KB）基于 conda 环境创建。由于后续训练微调需要利用 [LWFlow](https://github.com/OptimalScale/LMFlow)，安装 conda 环境的命令遵循 LWFlow 官方文档的指导，python 版本为 `Python 3.9.21`。运行命令如下：
```sh
git clone -b v0.0.9 https://github.com/OptimalScale/LMFlow.git
cd LMFlow
conda create -n kb-lmflow python=3.9 -y
conda activate kb-lmflow
conda install mpi4py -y
pip install -e .
```
项目的其他依赖配置在 `requirement.txt` 中记录，在创建并激活了 conda 环境 `kb-lmflow` 之后，执行命令 `pip install -r requirements.txt` 进行安装。

### 注意

LMFlow v0.0.9 默认下载的是 `transformers==4.47.0 (对应 tokenizers==0.21.0)`，这个版本过于新，如果需要测试一些比较旧的模型，需要手动降低 `transformers` 库的版本。

在本项目中，为了运行 `THUDM/chatglm2-6b`，需要在完成上述 LMFlow 环境配置的前提下，将

`THUDM/chatglm2-6b` 这个模型似乎不支持 transformers >= 4.42.0。测试过后，需要以下旧版本的包

```
transformers==4.41.0 (对应 tokenizers==0.19.1)
# 下面两个库和 LMFlow v0.0.9 所依赖的版本是一致的
torch==2.5.1
sentencepiece==0.2.0
```

## Task 2.1 探究大模型的知识边界 & 生成训练数据

本小组的该任务参考了 [R-Tuning](https://github.com/shizhediao/R-Tuning)。

以 “评估 `openlm-research/open_llama_3b` 模型在 `MMLU` 数据集上的知识边界” 为例，运行命令如下：
```sh
HF_ENDPOINT='https://hf-mirror.com' python run_MMLU.py --model openlm-research/open_llama_3b
```

生成的训练数据在项目根目录的 `traing_data`文件夹下。

****

探查大模型的“知识边界” 这一任务主要有三个方面需要探究：

### RQ1. 如何去定义模型的知识边界?

我们小组定义大模型的知识边界是：

1. 对于一个问题 $Q$，待评估的模型 $M$ 对该问题做出回答 $A'$ 。

2. 对比 Benchmark 标准答案 $A$ 和 模型的回答 $A'$ 。如果 $A'$ 和 $A$ 一致（或者相似度达到某个较高的阈值），那么认为 **模型的知识边界中包含该问题 $Q$** 。
3. 否则，则认为 **模型的知识边界中不包含该问题 $Q$** 。 $Q$ 在模型的知识边界之外。

一种比较朴素的定义方式是：基于某数据集，直接通过 Prompt 问大模型 $M$ 是否了解该知识。让大模型 $M$ 在自己难以回答该问题 $Q$ 时，**自己回答”自己不知道“**，以此来衡量该大模型 $M$ 在数据集 $D$ 所代表的领域上的知识边界。其 Prompt 可以定义为如下形式：

```
Here is a question for you:
[Question]
If you know the answer, say "I know" and answer the Question;
If you don't know how to solve the question, just say "I dont't know". 
Your answer is:
```

我们没有采用这种定义方式，主要基于三点考虑：

1. 首先，我们确实在和一些模型的交互中遇到了 **模型主动声称自己难以回答某个问题**，但是这种情况**并不能稳定复现**。很可能在数十次尝试中，才会出现这么一次。在经过训练后，绝大多数模型更倾向于 **执意给出不那么有把握的答案**，主动说明自己难以解决问题的情况非常稀少。
2. 如果将评估的权力交给模型自身，那么对于知识边界的衡量将完全是一个黑盒过程。模型声称自己能不能回答，它真的就不能回答吗？模型对于自己知识边界的评估，是我们无法验证，也无法量化的。
3. 在 Prompt 中对于”I Know“和”I Don't Know“的直接描述，有时候会影响模型产生摇摆不定的结果。可能将一些原本能回答的问题，改为了”I Don't Know“。

相比之下，采用和 Benchmark 对比的方式，至少在有精确答案的情况下，可以以量化的方式，相对明确地衡量模型的知识边界。

****

值得一提的是，在研究过程中，我们小组对于 **模型在做问答题，还是在做文本续写题** 产生了疑惑。

在 MMLU 数据集的测试过程中，我们注意到：当使用 `openlm-research/open_llama_3b` 和 `Qwen/Qwen-7B` 这样相对比较旧的 [Text Generation](https://huggingface.co/tasks/text-generation) 模型时，如果不控制其输出的 token 尺寸（即 `max_new_tokens` 参数），那么在其作为”回答“的字母（A、B、C、D）之后，往往会跟随大量的无意义的杂乱文字（通常和 fewshot 的内容近似甚至完全一致）。即使我们 **在 prompt 中要求其输出 选择该选项的理由**，在作为”回答“的字母之后，也会生成一些和 fewshot 相似的问答内容，而非理由。

![image](https://github.com/user-attachments/assets/f3dbd7fa-20b9-413d-92a3-d588138d0161)

在比较新的 Text Generation 模型（如 `Qwen/Qwen2.5-3B`）和 Chat 模型（如 `THUDM/chatglm2-6b`）上，通过 **在 prompt 中要求其输出 选择该选项的理由**，类似的问题没有出现。

因此，我们组有些怀疑，这些较旧的模型对于问题给出的回答，真的算是它们的”知识“吗？它们是 **在做问答题，还是在做文本续写题**？对于它们来说，我们用于评估知识边界的问题，是否只是一个用于续写文本的”前文环境“？如果模型并没有从”知识“的角度去理解问题的含义，那么像 [R-Tuning](https://github.com/shizhediao/R-Tuning) 等工作中使用的这种模型，探索它们的知识边界是否真的有意义呢？对于这些模型而言，它们有”知识“，但是是否拥有我们希望测试的那种”知识“呢？

不可否认的是，大语言模型的文本生成本身也是基于概率，有天然的不确定性。但是从宏观上来说，现代的模型基本上是 **将海量的客观知识在训练过程中存储在参数化记忆中**，然后基于用户输入的查询，通过 **模式识别和推理** 生成最终的输出。具体来说，模型可以通过各种方式（如注意力机制 Attention Mechanism）**对输入文本的不同部分进行加权处理，结合模型中存储的知识，生成与查询最相关的输出**。所以，对于上述模型而言，它们是基于”针对问题的所需知识“做出的回答，还是仅仅基于”问题的字面含义“做文本续写任务？

不过，从某种意义上说，也可以说我们在用这个问题去评估 **模型正确地做文本续写的能力边界**。而且本身也找不到很好的方式去 **区别衡量** 这两种能力。所以最终我们还是以相同的方式去评估了所有模型的知识边界。

### RQ2. 如何去评估模型的知识边界？

项目探寻了如下模型的知识边界：

```
openlm-research/open_llama_3b
THUDM/chatglm2-6b
Qwen/Qwen-7B
Qwen/Qwen2.5-3B
```

项目在不同的数据集上对模型的知识边界进行了评估【目前只完成了 MMLU 上的评测】。

```
MMLU
```

主要分为两大类型：有精确答案的数据集（如选择题）；开放域问答数据集。

#### 选择题任务——有明确且简短的答案对比

MMLU 数据集是一个单选题数据集，涵盖了 28 个领域，总计有 2448 个问题。每个问题有四个选项以供选择，并附有标准答案。MMLU 数据集有 **领域信息**，方便我们去 **描述大模型在不同领域的知识边界**。

在参考 [R-Tuning](https://github.com/shizhediao/R-Tuning) 的基础上，我们的评估方式为：

1. 基于 MMLU 数据集，构建 5-shots 问题 prompt 或者 one-shot 问题 prompt 作为模型的输入，在得到每个模型对于问题的回答输出后，经过一些处理，**检查标准答案所在的选项是否在模型的输出中**。
2. 计算 **模型回答正确/错误的比例**。一个直观的解释是，待评估的模型 $M$ 在 x 领域的问答上正确率高，说明模型 $M$ 对 x 领域的知识掌握程度高，模型知识边界更广。

#### 自然语言问答任务——需要语义化对比 $A'$ 和 $A$

对于选择题任务的评估，在校验时只需要采用基于字符匹配的方法；而对于自然语言问答任务，**R-Tuning 这篇论文采用的方法过于简单粗暴**，产生了很多 **不太正确的知识边界评估**。而基于在这种情况下构建的 **拒绝意识指令数据集 Refusal-Aware Instruction Dataset **去微调模型，很明显会产生 **过度拒绝 over-refusal**。

> R-Tuning 中的相关逻辑代码如下：
>
> ```python
> if sample[1] in output:
>     text += " I am sure."
> else:
>     text += " I am not sure."
> ```

尽管可以用 **模型回答的忠诚度** 来解释较为严格的评估范式的合理性，但是完全按照字符匹配的方式，我们小组还是认为不能接受，这里举一个例子：

> 在 `ParaRel` 数据集上有一个问题：`In what field does Anaxagoras work?`
>
> 数据集的 Ground-Truth 回答 $A$ 是 `philosophy`；而 `openlm-research/open_llama_3b` 的回答 $A'$ 是 `Anaxagoras was a Greek philosopher who lived in the...`。诚然，相比标准答案的单独一个单词，模型的回答加入了额外的信息，可能会偏离用户的意图。**模型回答的忠诚度确实相对较低**。
>
> 但是，对于“领域”和“领域对应的职业”这种 **同源派生词的不同**，我们认为 **不能只用简单的字符匹配来判断模型对于该问题就是不知道的**。
>
> 我们向论文作者提交了 [Git repo Issue](https://github.com/shizhediao/R-Tuning/issues/11)，作者以 希望实现更好的效果 和 模型回答的忠诚度 回答。

**我们对此提出了 2 个改进方向**，并做了相应分析：

1. 引入一个评判模型 $M\_{judge}$（如 `GLM-4-Air`）以衡量 **Benchmark 标准答案 $A$**   和  **模型的回答 $A'$** 的一致性。利用大模型强大的语言理解能力，对自然语言形式的两个回答进行比较。具体实现上，可以让 $M\_{judge}$ 先判断两个回答的一致性 `YES/NO`，然后给出判断一致性的理由，之后 **统计 `YES/NO` 的比例**。
   - 优点是简单易行，且模型的判断理由是可以看到的。
   - 缺点是模型输出的不稳定性，可能多次评判的结果并不相同，也会随着 prompt 的影响而改变。但是，**肯定要比原始的方法要好**。
2. 对两个回答进行 **文本编码**，计算嵌入向量的相似度，或者是别的衡量标准，例如 BERT、TF-IDF。
   - 优点是在确定了嵌入模型等工作流后，**评判结果是确定的。**
   - 缺点是文本编码相对黑盒，有些时候，即使文本相似度高，结果也并不能保证是对的。而且很多情况下，模型的回答 $A'$ 是一个句子，标准回答 $A$ 是一个单词或者很短的句子。用基于文本嵌入的方式去衡量，相似度的 **阈值难以去设置**，且必然会随着数据集的不同而波动。

最终，我们尝试使用了第一种方式。

在评判过程中，因为我们定义的知识边界是 **以数据集的标准答案为基准**，所以需要摒弃掉评判模型 $M\_{judge}$（如 `GLM-4-Air`）自身所知的客观事实，严格按照 Benchmark 去评判模型的回答。

>以 `What field does Alan Tuning work in?` 为例
>
>数据集的 Ground-Truth 回答 $A$ 是 `logic`，而模型的回答 $A'$ 是 `Alan Tuning is a computer science`。尽管从客观事实来看，两个回答均正确，也不可能说模型不知道 Alan Tuning 的工作领域。但是从 **模型回答的忠诚度** 的角度出发，`computer science` 和 `logic` 的差异相对较大，且两者不存在从属关系，所以为了避免评判模型 $M\_{judge}$ 基于它自己的客观事实，我们会在 prompt 设计和参数设置上做出相应要求。
>
>![image](https://github.com/user-attachments/assets/741f6035-d254-4cc5-9e50-e6cf2f40a57c)

****

值得一提的是（梅开二度），在研究过程中，我们小组对于 **数据集本身的 正确性** 产生了疑惑。

数据集中一些问题和答案的设置，显得有些 **限制性过强，过于刻意体现回答忠诚度**

![image](https://github.com/user-attachments/assets/00164615-f7eb-4d81-90d3-aef176cde2e6)

Bruce Perens 是知名的美国软件自由活动家和开源软件的倡导者，是 Debian Linux 发行版的关键创始人之一。所以模型的回答 $A'$ 是对的，评判模型 $M\_{judge}$ 的判断也是合理的，程序员确实是软件领域中的一种职业。但是 Benchmark 中强调 Bruce Perens 是一个程序员，可能会存在一些模糊。

有时，数据集的回答只给定了问题答案中的某一个领域，但是当模型的回答是 **另一个领域且正确** 的时候，难以根据数据集去判断模型是错误的。

第一个问题是 问图灵的领域：`What field does Alan Turing work in?`。这个问题在数据集里面出现了两次，一次是 `logic`， 一次是 `mathematics`。但是我们测试的多个模型都回答：图灵是一个 `computer scientist`，从客观事实上来看，不能说模型回答的是错误的，也不能说模型不知道这个问题，它 **属于模型的知识边界的内外的哪一边呢**？—— 又就回到了最开始的问题：**如何去定义知识边界**。

也许，这就只能是用 **模型回答忠诚度** 在不同研究者的理解不同来解释了。

### RQ3. 模型知识边界的评估结果

【这块我之后整个表格来呈现。评估结果目前是以 json 形式】



## Task 2.2.1 让大模型在不知道的时候回答“不知道”


## Task 2.2.2 避免知道的知识被错回复为“不知道”，降低或者避免over-refusal
[TASK2.2.2 理论](./LMFlow/Task2_2_2.md)
进行finetune时，为避免模块冲突，请用conda复制一份环境，并还原`transformers == 4.47.0`和`tokenizers==0.21.0`版本
其他修正方式请参照[finetune管理文档](./LMFlow/FinetuneChange.md)