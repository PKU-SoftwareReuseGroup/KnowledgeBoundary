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

### 补充

后续发现 LMFlow 的微调效果不好，又加入了 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)。

## Task 2.1 探究大模型的知识边界 & 生成训练数据

本小组的该任务参考了 [R-Tuning](https://github.com/shizhediao/R-Tuning)。

以 “评估 `openlm-research/open_llama_3b` 模型在 `MMLU` 数据集上的知识边界” 为例，运行命令如下：
```sh
HF_ENDPOINT='https://hf-mirror.com' python run_MMLU.py --model openlm-research/open_llama_3b
```

生成的训练数据在项目根目录的 `traing_data`文件夹下。

****

探查大模型的 “知识边界” 这一任务主要有三个方面需要探究：

#### RQ1. 在评估场景下，如何去定义模型的知识边界?

我们小组认为：在评估场景下，大模型的知识边界是：

1. 对于一个问题 $Q$，待评估的模型 $M$ 对该问题做出回答 $A'$ 。

2. 对比 Benchmark 中的标准答案 $A$ 和 模型的回答 $A'$ 。如果 $A'$ 和 $A$ 一致（或者相似度达到某个较高的阈值），那么认为 **模型的知识边界中包含该问题 $Q$** 。
3. 否则，则认为 **模型的知识边界中不包含该问题 $Q$** 。 $Q$ 在模型的知识边界之外。

另外一种比较朴素的定义方式是：基于某数据集，直接通过 Prompt 问大模型 $M$ 是否了解该知识。让大模型 $M$ 在难以回答该问题 $Q$ 时，**自己回答：”我不知道“**，以 **大模型愿意回答的问题** 来衡量该大模型 $M$ 在数据集 $D$ 所代表的领域上的知识边界。其 Prompt 可以定义为如下形式：

```
Here is a question for you:
[Question]
If you know the answer, say "I know" and answer the Question;
If you don't know how to solve the question, just say "I dont't know". 
Your answer is:
```

我们没有采用这种定义方式，主要基于三点考虑：

1. 首先，我们确实在和一些模型的交互中遇到了 **模型主动声称自己难以回答某个问题**，但是这种情况**并不能稳定复现**。在经过预训练后，绝大多数模型更倾向于 **执意给出不那么有把握的答案**，主动说明自己难以解决问题的情况非常稀少。
2. 如果将评估的权力交给模型自身，那么对于知识边界的衡量将完全是一个黑盒过程。模型声称自己能不能回答，它真的就不能回答吗？模型对于自己知识边界的评估，是我们无法验证，也无法量化的。
3. 在 Prompt 中对于”I Know“和”I Don't Know“的直接描述，有时候会影响模型产生摇摆不定的结果。可能将一些原本能回答的问题，改为了”I Don't Know“。

相比之下，采用和 Benchmark 对比的方式，至少在有精确答案的情况下，可以以量化的方式，相对明确地衡量模型的知识边界。

****

值得一提的是，在研究过程中，我们小组对于 **模型在做问答题，还是在做文本续写题** 产生了疑惑。

在 MMLU 数据集的测试过程中，我们注意到：当使用 `openlm-research/open_llama_3b` 和 `Qwen/Qwen-7B` 这样相对比较旧的 [Text Generation](https://huggingface.co/tasks/text-generation) 模型时，如果不控制其输出的 token 尺寸（即 `max_new_tokens` 参数），那么在其作为”回答“的字母（A、B、C、D）之后，往往会跟随大量的无意义的杂乱文字（通常和 fewshot 的内容近似甚至完全一致）。即使我们 **在 prompt 中要求其输出 选择该选项的理由**，在作为”回答“的字母之后，也会生成一些和 fewshot 相似的问答内容，而非理由。

![image](https://github.com/user-attachments/assets/f3dbd7fa-20b9-413d-92a3-d588138d0161)

在比较新的 Text Generation 模型（如 `Qwen/Qwen2.5-3B-Instruct`）和 Chat 模型（如 `THUDM/chatglm2-6b`）上，通过 **在 prompt 中要求其输出 选择该选项的理由**，类似的问题没有出现。

因此，我们组有些怀疑，这些较旧的模型对于问题给出的回答，真的算是它们的”知识“吗？它们是 **在做问答题，还是在做文本续写题**？对于它们来说，我们用于评估知识边界的问题，是否只是一个用于续写文本的”前文环境“？如果模型并没有从”知识“的角度去理解问题的含义，那么像 [R-Tuning](https://github.com/shizhediao/R-Tuning) 等工作中使用的这种模型，探索它们的知识边界是否真的有意义呢？对于这些模型而言，**它们拥有的”知识“是否是我们希望测试的那种”问答知识“呢**？

不可否认的是，大语言模型的文本生成本身也是基于概率，有天然的不确定性。但是从宏观上来说，现代的模型基本上是 **将海量的客观知识在训练过程中存储在参数化记忆中**，然后基于用户输入的查询，通过 **模式识别和推理** 生成最终的输出。具体来说，模型可以通过各种方式（如注意力机制 Attention Mechanism）**对输入文本的不同部分进行加权处理，结合模型中存储的知识，生成与查询最相关的输出**。所以，对于上述模型而言，它们是基于”解决问题所需的知识“做出的回答，还是仅仅基于”问题的字面含义“做文本续写任务？

不过，从某种意义上也可以认为：我们在用数据集中的问题去评估 **模型正确地做文本续写的能力边界**。而且本身也找不到很好的方式去 **区别衡量** 这两种能力。所以最终我们还是以相同的方式去评估了所有模型的知识边界。

### RQ2. 如何去评估模型的知识边界？

项目探寻了如下模型的知识边界：

```
openlm-research/open_llama_3b
THUDM/chatglm2-6b
Qwen/Qwen-7B
Qwen/Qwen2-1.5B-Instruct
Qwen/Qwen2.5-3B
Qwen/Qwen2.5-3B-Instruct
```

项目在不同的数据集上对模型的知识边界进行了评估，主要分为两大类型：有精确答案的数据集（单项选择题）；开放域问答数据集。

#### 选择题任务——有明确且简短的答案对比

##### 数据集：MMLU

[MMLU](https://github.com/hendrycks/test) 数据集涵盖 STEM、人文（humanities）、社会科学（social sciences）和 其他（others）等领域的 57 个学科（subject）。 它的难度从初级到高级，既考验世界知识，又考验解决问题的能力。 学科范围从数学和历史等传统领域到法律和伦理等更为专业的领域。学科的粒度和广度使该基准成为识别模型盲点的理想选择。

MMLU 数据集是一个单选题数据集，每个问题有四个选项。MMLU 共收集了 15908 个问题，并将其分为 few-shot 开发集、验证集和测试集。few-shot开发集每个学科有 5 个问题，验证集可用于选择超参数，由 1540 个问题组成，测试集有 14079 个问题。 每个学科至少包含 100 个测试问题。

##### 数据集：C-Eval

[C-Eval](https://github.com/hkust-nlp/ceval) 数据集涵盖了 52 个不同学科的 13948 个多项选择题，分为四个难度级别。

C-Eval 数据集是一个单选题数据集，每个问题有四个选项。**它是中文数据集，对于国内模型的评测友好。**

> 这里附上一个很有价值的博客：[NLP 大模型探索：MMLU数据集评测 ](https://percent4.github.io/NLP（七十八）大模型探索：MMLU数据集评测/)

****

MMLU 数据集 和 C-Eval 数据集有 **领域信息**，方便我们去 **描述大模型在不同领域的知识边界**。

在参考 [R-Tuning](https://github.com/shizhediao/R-Tuning) 的基础上，我们的评估方式为：

1. 基于数据集给的 few-shots（如 MMLU 的 `MMLU_ID_prompt.json`；把 C-Eval 的 `val` 用作 prompts），构建 5-shots 问题 prompt 作为模型的输入。在得到每个模型对于问题的回答输出后，经过一些处理，**检查标准答案所在的选项是否在模型的输出中**。
   - 对于大部分模型，特别是 **需要进行 2.2.1 和 2.2.2 训练的模型**，模型答案的确定方法是：判断 LLM 后续 token 为 A, B, C 或 D 的概率，选择其中 **token 生成概率最高的选项** 作为模型的输出。
   - `THUDM/chatglm2-6b` 没有对于概率生成方式的官方支持，`Qwen/Qwen2.5-3B` 实在是 **没法调整成直接输出字母答案**，所以这两个模型只能采用字符串截取，因此无法参与后续训练。
2. 计算 **模型回答正确/错误的比例**。一个直观的解释是，待评估的模型 $M$ 在 x 领域的问答上正确率高，说明模型 $M$ 对 x 领域的知识掌握程度高，模型知识边界更广。

#### 开放域问答任务——需要语义化对比 $A'$ 和 $A$

##### 数据集：ParaRel

涵盖对于名人工作领域等的自然语言问答（英文）。

****

对于选择题任务的评估，在校验时只需要采用基于字符匹配的方法；而对于开放域问答任务，**R-Tuning 这篇论文采用的方法过于简单粗暴**，产生了很多 **不太正确的知识边界评估**。而基于在这种情况下构建的 **拒绝意识指令数据集 Refusal-Aware Instruction Dataset **去微调模型，很明显会产生 **过度拒绝 over-refusal**。

> R-Tuning 中的相关逻辑代码如下：
>
> ```python
> if sample[1] in output:
>     text += " I am sure."
> else:
>     text += " I am not sure."
> ```

尽管可以用 **模型回答的忠诚度** 来解释较为严格的评估范式的合理性，但是，我们小组还是认为 **完全按照字符匹配的方式难以接受**。这里举一个例子：

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
><img src="https://github.com/user-attachments/assets/741f6035-d254-4cc5-9e50-e6cf2f40a57c" alt="image" style="zoom:60%;" />

****

值得一提的是（梅开二度），在研究过程中，我们小组对于 **数据集本身的 正确性** 产生了疑惑。

数据集中一些问题和答案的设置，显得有些 **限制性过强，过于刻意体现回答忠诚度**

![image](https://github.com/user-attachments/assets/00164615-f7eb-4d81-90d3-aef176cde2e6)

Bruce Perens 是知名的美国软件自由活动家和开源软件的倡导者，是 Debian Linux 发行版的关键创始人之一。所以模型的回答 $A'$ 是对的，评判模型 $M\_{judge}$ 的判断也是合理的，程序员确实是软件领域中的一种职业。但是 Benchmark 中强调 Bruce Perens 是一个程序员，可能会存在一些模糊。

有时，数据集的回答只给定了问题答案中的某一个领域，但是当模型的回答是 **另一个领域且正确** 的时候，难以根据数据集去判断模型是错误的。

第一个问题是 问图灵的领域：`What field does Alan Turing work in?`。这个问题在数据集里面出现了两次，一次是 `logic`， 一次是 `mathematics`。但是我们测试的多个模型都回答：图灵是一个 `computer scientist`，从客观事实上来看，不能说模型回答的是错误的，也不能说模型不知道这个问题，它 **属于模型的知识边界的内外的哪一边呢**？—— 又就回到了最开始的问题：**如何去定义知识边界**。

也许，这就只能是用 **模型回答忠诚度** 在不同研究者的理解不同来解释了。

### RQ3. 模型知识边界的评估结果

下表只记录整个数据集的通过率，不区分子领域（但是已经分领域生成了 json 形式）。如需要请自行计算。

|         模型          | MMLU 通过率% |  C-Eval 通过率%  | ParaRel 通过率/判断通过率% |
| :-------------------: | :----------: | :--------------: | :------------------------: |
|    `open_llama_3b`    |    24.43     | (不具备中文能力) |        38.55/40.97         |
|     `chatglm2-6b`     |    39.17     |        -         |             -              |
|       `Qwen-7B`       |    21.41     |        -         |             -              |
| `Qwen2-1.5B-Instruct` |    50.08     |      68.35       |             -              |
|     `Qwen2.5-3B`      |    58.99     |        -         |          40/43.35          |
| `Qwen2.5-3B-Instruct` |    60.09     |      72.14       |             -              |





## Task 2.2.1 让大模型在不知道的时候回答“不知道”

我们通过 **拒绝意识指令集微调 RAIT**，使得模型具备拒绝能力。

要达到的目标是：**无需 Prompt 显式提示模型**（如加入：`If you don't know, please output 'I don't know'.`），Prompt 中仅有待回答选择题的题干和四个选项的时候，也可以让模型输出拒绝。

### 模型选择：

**Qwen2-1.5B-Instruct**

Qwen2.5-3B

openllama-3B

### 数据集选择：

单选题数据集：MMLU

测试微调模型**拒绝**能力：MMLU_ID_train

测试微调模型**泛化**能力：MMLU_ID_test、MMLU_OOD_test

### 微调方式：

使用`LLaMa-Factory`方式进行微调：

`LLaMa-Factory`微调需要使用如下的`alpaca`数据格式，

```json
[
  {
    "instruction": "用户指令（必填）",
    "input": "用户输入（选填）",
    "output": "模型回答（必填）",
    "system": "系统提示词（选填）",
    "history": [
      ["第一轮指令（选填）", "第一轮回答（选填）"],
      ["第二轮指令（选填）", "第二轮回答（选填）"]
    ]
  }
]
```

所以在对1.0所做的知识边界测试中，同步生成了该类型的数据集：

如果第一步知识边界测试中，模型回答对了问题，则保留其选择的正确答案，如果模型回答错，就用N替换其原本的答案，用意是用`N`来表示模型所不知道的知识，从而构建起模型的拒绝域。选择用`N`的理由是我们在面对选择题时，采用的方法是截取第一个`token`，并且根据其概率，选择概率最大的进行输出。在构建拒绝方式的时候输出`N`和`I don't know`本质上并没有区别，均可以表达模型具备了拒绝能力。在构建数据集时，我们选择了两种`Instruction`的构造方式：

**第一种**

本种方式更像是应用prompt显式提示大模型遇到不知道的问题就输出`N`。后续实验结果表明，将这种方式应用在`Instruction`中，可能模型理解的并不好，没有判断出哪些知识是自己掌握得并不好的。

```json
[
	  {
        "instruction": "If you know what the answer is, just print the answer, not the content of the answer.If you're not sure about the generated answer or you don't know how to answer the question, output: N.",
        "input": "Statement 1 | Every solvable group is of prime-power order. Statement 2 | Every group of prime-power order is solvable.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer:",
        "output": "N"
    },
    {
        "instruction": "If you know what the answer is, just print the answer, not the content of the answer.If you're not sure about the generated answer or you don't know how to answer the question, output: N.",
        "input": "Find all c in Z_3 such that Z_3[x]/(x^3 + cx^2 + 1) is a field.\nA. 0\nB. 2\nC. 1\nD. 3\nAnswer:",
        "output": "B"
    }
]
```

**第二种**

本种方式类似于打标签的方式，即在`Instruction`中说明，现在的`output`为`N`的是你所不确定的知识，现在的`output`为四个选项其中一个的为你所确定的知识，这样的`Instruction`构建更能够让模型清楚自己哪些知识是不清楚的，也能够建立起不确定知识输出`N`的拒绝意识，在后边的实验结果中显示，本种`Instruction`构建方式有效。

```json
[
    {
        "instruction": "Output as N means the knowledge you are not sure about, and output as one of A, B, C, D means the knowledge you are certain about.",
        "input": "Statement 1 | Every solvable group is of prime-power order. Statement 2 | Every group of prime-power order is solvable.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer:",
        "output": "N"
    },
    {
        "instruction": "Output as N means the knowledge you are not sure about, and output as one of A, B, C, D means the knowledge you are certain about.",
        "input": "Find all c in Z_3 such that Z_3[x]/(x^3 + cx^2 + 1) is a field.\nA. 0\nB. 2\nC. 1\nD. 3\nAnswer:",
        "output": "B"
    },
]
```

### 微调结果：

#### Qwen2.5-3B

该模型微调之后可以规整的输出选择题的答案，并且出现了少量的拒绝，能够证明我们的微调方法是有效的。

```json
    "Final_Evaluation": {
        "Pass": 1604,
        "UNPASS": 817,
        "REFUSE": 27,
        "Total": 2448,
        "Accuarcy": 0.6552
    }
```

但是其在未被微调之前并不能直接规整输出A，B，C，D四个选项其中的一个，而是只能输出：

>The answer is **A**. The statement is ……

由于我们下一步减少`over-refusal`的时候需要采样输出答案的概率，从而判断准确性和正确性，但是该模型无法采样到概率，所以不能应用在下一个减少`over-refusal`的任务当中。

#### OpenLLaMa-3B

该模型为**续写模型**，其并不能有效理解`prompt`的语义，在初始测试知识边界的环节，其输出和`prompt`提示语句`If you don't know, please output 'N'.`的位置有关，若放在待回答问题的前边，模型会忽略该`prompt`的存在，若放在待回答问题后，模型会根据该`prompt`续写，输出无意义和逻辑的回答。并不能规整的输出拒绝，在经过微调后，若不加入`If you don't know, please output 'N'.`，仍不能输出有效的拒绝，即在答案中并没有`N`的出现，若在此时加入`prompt`，无论加在什么位置，模型均可以输出拒绝`N`。不过据观察，此时模型的回复中不再出现答案`D`，应该是模型的能力所限。

#### Qwen2-1.5B-Instruct

由于该模型是可以理解`Instruct`内容的，即如果`prompt`中提示`If you don't know, please output 'N'.`，模型可以在面对问题时输出拒绝。所以使用`prompt`方式来显式提示模型遇到不知道的问题就输出拒绝，是会影响判断模型微调前后的结果的，所以在微调前后的测试中，`prompt`中只有问题的题干和四个选项。

在测试中发现，使用上边第一种`Instruction`方式构建的数据集效果并不理想，在MMLU_ID_train的2448个问题中只出现了11个拒绝。

```json
    "Final_Evaluation": {
        "Pass": 1395,
        "UNPASS": 1042,
        "REFUSE": 11,
        "Total": 2448,
        "Accuarcy": 0.5699
    }
```

下图为使用第二种`Instruction`进行微调，模型微调前后在MMLU_ID_train数据集的各个子领域中的表现对比：

![Before_ft](./0_GraphDisplay/res/2.2.1/Before_ft.png)

![After_ft](./0_GraphDisplay/res/2.2.1/After_ft.png)

从对比图中可以看出，微调后的模型不仅出现了拒绝回答问题的情况，成功构建了拒绝域，而且对于大部分的错误域问题都可以成功拒绝，只有一小部分原本正确的答案出现了`over-refusal`的问题。

此外，为了测试微调后的模型在未经训练过的数据及域外知识上的表现，测试模型的泛化能力，我们还在MMLU_ID_test和MMLU_OOD_test数据集上进行了实验：

![b74752d5bfeb2e943f278072f833b38](./0_GraphDisplay/res/2.2.1/b74752d5bfeb2e943f278072f833b38.png)

测试结果表明，在未经训练过的数据上仍然可以构建起拒绝域，说明模型在面对不确定的问题时，已经有了拒绝意识，可以在一定的程度上减少模型的幻觉问题。

在三种模型上的微调实验结果纵向对比表明，微调的效果与模型自身的能力息息相关。在Qwen2-1.5B-Instruct上的实验表明，我们所提出的使用第二种`Instruction`构建的数据集进行微调的方式是有效的，可以成功让微调模型在无prompt提示的情况下输出拒绝，成功构建拒绝域。并且仅有少量的`over-refusal`问题出现。


## Task 2.2.2 避免知道的知识被错回复为“不知道”，降低或者避免over-refusal
进行finetune时，为避免模块冲突，请用conda复制一份环境，并还原`transformers == 4.47.0`和`tokenizers==0.21.0`版本
其他修正方式请参照[finetune管理文档](./LMFlow/FinetuneChange.md)

先下载llama3b 模型

```
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download openlm-research/open_llama_3b --local-dir ../../models/openlm-research/open_llama_3b
```

注意在相应文件中 **修改model和dataset路径**

```bash
#文件夹 1_Qwen2-1.5B-Instruct_{dataset}/ 中存放相关代码

python 1_run_MMLU_RAIT #用MMLU数据集对llama3b模型进行提问并统计ACC，同时计算每一个问题的COR和CER

#将产生的问答结果用 LLaMa-Factory 微调模型 

python 2_run_MMLU_finetune.py #用MMLU数据集对微调后的模型进行提问并统计ACC，同时计算每一个问题的COR和CER

python 3_run_select_dataset.py #筛选数据集

#将筛选后的结果用 LLaMa-Factory 微调模型 

python 4_run_MMLU_finetune #统计 筛选数据集 前后 正确个数、正确个数提升、正确个数下降、正确率提升、确定性提升等

#统计数据以及画图代码 存放于 0_GraphDisplay/ 文件夹中

```


### 拒绝感知指令调优（RAIT）
**Cor-RAIT：** 用原始数据集向语言模型询问，根据语言模型的回答将数据集进行分类： $D_{rait} = D_{van} \cup D_{idk}$ ，其中 $D_{idk}$ 是将答案修改为"I don't know"的问题集。 $D_{van}$ 是回答正确的问题集。

## over-refusal的两种情况：

**Static conflict**   当在大语言模型的特征空间内的相似样本接收到不同的监督信号（原始的与修改后的 “我不知道”）时，就会发生静态冲突。 
在LLM特征空间中，在 `Cor-RAIT` 框架下，可以将两个紧密定位（相似）的样本分配给 $D_{van}$ 和 $D_{idk}$ ，这些相似的样本在训练期间提供了冲突的监督标签信号，削弱了LLM区分已知和未知问题的能力，导致过度拒绝。

**原因**：是两类问题在样本空间中的余弦相似度接近。文章中用  相似样本冲突率 $CRSS$ 来衡量**静态冲突率**，较高的CRSS表明存在更多冲突的相似样本对，可能导致过度拒绝。

$$
CRSS=\frac{\sum_{x \in D_{idk}}1(max_{x_j \in D_{van}}cos(r_i,r_j)>\tau_{sim})}{|D_{idk}|}
$$

**解决方法**：文章只说了采用Cor-Cer-RAIT之后 $CRSS$ 减小，得知静态冲突变小。

**Dynamic conflict**  在监督微调（SFT）过程中，语言模型（LLM）的知识不断演化，从而能够回答之前无法回答的问题。然而，这些现在能够回答的训练样本仍然保留着基于初始语言模型状态的 “我不知道” 监督信号，进而导致了不一致的情况。 
训练过程中忽略了LLM的知识状态的动态变化，研究 (Ren et al. 2024; Ren and Sutherland 2024) 表明 LLM 的知识状态在监督微调 (SFT) 期间发生变化，问题可能从未知转移到已知，反之亦然。

**解决方法：**

​	对于**正确性**，去掉 $D_{idk}$ 中 **微调前后模型** 回答问题中 **从错误回答变成正确回答** 的部分

​	对于**确定性**，去掉 $D_{idk}$ 中 **高确定性** 部分问题和 $D_{van}$ 中 **低确定性** 的部分问题

**感知流** 就是 **正确性** 和 **确定性** 的变化

### 在MCQA（知识导向的多项选择题问答）任务中，选择 MMLU 数据集
**正确性correctness:**
$$
Cor(\mathcal{M},x_i)=p(x_i.a|x_i.q,\mathcal{M})
$$

**确定性Certainty：**

$$
Cer(\mathcal{M},x_i)=-\sum_{\hat{a}\in O}P(\hat{a}|x_i.q,\mathcal{M})log(p(\hat{a}|x_i.q,\mathcal{M}))
$$

在本次任务中，对MMLU进行询问模型时：
```python
 probs = (
    torch.nn.functional.softmax(
        torch.tensor(
            [
                logits[tokenizer("A").input_ids[0]],
                logits[tokenizer("B").input_ids[0]],
                logits[tokenizer("C").input_ids[0]],
                logits[tokenizer("D").input_ids[0]],
            ]
        ),
        dim=0,
    )
    .detach()
    .cpu()
    .numpy()
)
```
该函数获知对于 `A` `B` `C` `D`四个答案概率，由此通过
```python
CORCER[i][sample[0]]["COR"] = probs[np.argmax(probs)].astype(float) 
```
来获知对于正确答案的概率 $Cor(\mathcal{M},x_i)$
通过
```python
#统计问题的精确度,注意将float32转化为float,不然JSON不支持
np_probs = np.array(probs)
log_probs = np.log(np_probs)
CORCER[i][sample[0]]["CER"]=-np.sum(np_probs*log_probs).astype(float)
```
来获知对于每个问题答案 $x_i$ 的确定性 $Cer(\mathcal{M},x_i)$
### 在OEQA（开放式回答）任务中，选择 TriviaQA 数据集
**正确性correctness:**

$$
Cor(\mathcal{M},x_i)=\frac{1}{N}\sum_{\hat{a_j}\in \hat{A}_{i}}1(\hat{a_j}=x_i.a)
$$

**确定性Certainty：**

$$
Cer(\mathcal{M},x_i)=\frac{1}{N(N-1)}\sum_{\hat{a_j},\hat{a_k}\in \hat{A}_{i},j\ne k}\cos(E(a_j),E(a_k))
$$

其中 $\hat{A_i}$ 是对于问题 $i$ 的多次回答答案的集合，  $E(a_j)$ 是回答 $a_j$ 的表征向量。

表征向量由 https://huggingface.co/sentence-transformers/all-MiniLML6-v2 模型算出

## 方法：

##### Stage 1: Query the Knowledge State and Flow of LLM  	

首先，执行知识状态查询，以获得模型对源数据集中样本的响应的正确性和确定性。然后，对模型进行预演训练，从而得到被扰动的版本。通过比较扰动前后的知识状态，我们推导出在有监督的微调过程中知识流的指标。

##### Stage 2: Refusal-Aware Instructions Construction and Tuning  	

使用第一阶段的知识状态和流程，我们从 $D_{src}$ 中选择合适的样品来构建 RAIT数据，用于微调初始模型。

## 原模型测试参数设置：

用 MMLU 数据集询问原模型，前加5个 `fewshot`， `message`设置如下：

```python
messages = [
    {"role": "system", "content": f"You are an expert on {subject}. You must just choose the answer."},
    {"role": "user", "content": full_input}
]
```

生成两个文件 `MMLU.json` 和 `MMLU_LF.json`: `MMLU.json`是包含5 `fewshot` 的结果，供 `Lmflow` 微调模型；`MMLU_LF.json` 是不包含 `fewshot` 的数据集，供LLaMa-Factory使用。
其中 `MMLU_LF.json` 参数设置如下:

```json
[
    {
        "instruction": "Output as N means the knowledge you are not sure about,and output as one of A, B, C, D means the knowledge you are certain about.",
        "input": "Statement 1 | Every solvable group is of prime-power order. Statement 2 | Every group of prime-power order is solvable.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer:",
        "output": "N"
    },
]
```

## 微调后的参数设置

用 MMLU 数据集询问原模型，基本不给任何提示信息, message设置如下：

```python
messages = [
    {"role": "system", "content": f"You are an expert on {subject}."},
    {"role": "user", "content": full_input}
]
```

在模型生成结果设置中 **增加对于 N 的概率计算**：

```python
probs = (
    torch.nn.functional.softmax(
        torch.tensor(
            [
                logits[tokenizer("A").input_ids[0]],
                logits[tokenizer("B").input_ids[0]],
                logits[tokenizer("C").input_ids[0]],
                logits[tokenizer("D").input_ids[0]],
                # FIX
                logits[tokenizer("N").input_ids[0]],
            ]
        ),
        dim=0,
    )
    .detach()
    .cpu()
    .numpy()
)
```

## 测试结果

#### 模型 `Qwen2-1.5B-Instruct` 模型 在 `MMLU` 数据集上的表现：

```json
//原始模型
"Final_Evaluation": {
    "Pass": 1226,
    "Total": 2448,
    "Accuarcy": 0.5008
}
//微调模型
"Final_Evaluation": {
    "Pass": 1091,
    "UNPASS": 360,
    "REFUSE": 997,
    "Total": 2448,
    "Accuarcy": 0.4457
}
//筛选数据集，保留度0.5
"Final_Evaluation": {
    "Pass": 1059,
    "UNPASS": 525,
    "REFUSE": 864,
    "Total": 2448,
    "Accuarcy": 0.4326
}
//筛选数据集，保留度0.2
"Final_Evaluation": {
    "Pass": 1139,
    "UNPASS": 874,
    "REFUSE": 435,
    "Total": 2448,
    "Accuarcy": 0.4653
}
```
原始模型

![image](./0_GraphDisplay/res/qwen2-1-5B-MMLU/Origin%20Model.png)

RAIT 微调后模型

![image](./0_GraphDisplay/res/qwen2-1-5B-MMLU/First%20finetuned%20Model.png)

数据集过滤后，保存 50% 的效果。可以看到黄色的拒绝域减少了，绿色的正确域增加了或者不变，说明 **模型原先知道的，被 RAIT 训练 导致误答“不知道”的情况减少了**：

![image](./0_GraphDisplay/res/qwen2-1-5B-MMLU/The%20dataset%20was%20filtered%20with%20a%20threshold%20of%200.5.png)

数据集过滤后，保存 20% 的效果。可以看到，**过度拒绝的现象缓解程度非常明显**：

![image](./0_GraphDisplay/res/qwen2-1-5B-MMLU/The%20dataset%20was%20filtered%20with%20a%20threshold%20of%200.2.png)

总的结果如下：

![image](./0_GraphDisplay/res/qwen2-1-5B-MMLU/Total%20Eva.png)

#### 对于模型 qwen2-1.5B-Instruct 模型 在 `CEval` 数据集上的表现：

原始模型

![image](./0_GraphDisplay/res/qwen2-1-5B-CEval/01_Origin.png)

RAIT 微调后模型

![image](./0_GraphDisplay/res/qwen2-1-5B-CEval/02_FirstfinetuneModel.png)

数据集过滤后，保存 50% 的效果。可以看到黄色的拒绝域减少了，绿色的正确域增加了或者不变，说明 **模型原先知道的，被 RAIT 训练 导致误答“不知道”的情况减少了**：

![iamge](./0_GraphDisplay/res/qwen2-1-5B-CEval/04_The%20dataset%20was%20filtered%20with%20a%20threshold%20of%200.5.png)

数据集过滤后，保存 20% 的效果。不幸的是：<font color='red' size='4px'>**缓解过度拒绝使得模型失去了拒绝能力**：</font>

![image](./0_GraphDisplay/res/qwen2-1-5B-CEval/03_The%20dataset%20was%20filtered%20with%20a%20threshold%20of%200.2.png)

总的结果如下：

![image](./0_GraphDisplay/res/qwen2-1-5B-CEval/05_output.png)
