# 计算语言学大作业：探寻知识边界

## 运行项目

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

项目探寻了如下模型的知识边界：
```
openlm-research/open_llama_3b
THUDM/chatglm2-6b
Qwen/Qwen-7B
```

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

## Task 2.2.1 让⼤模型在不知道的时候回答“不知道”


## Task 2.2.2 避免知道的知识被错回复为“不知道”，降低或者避免over-refusal
### 拒绝感知指令调优（RAIT）
**Cor-RAIT：**用原始数据集向语言模型询问，根据语言模型的回答将数据集进行分类：$`D_{rait} = D_{van} \cup D_{idk}`$，其中$`D_{idk}`$是将答案全部修改为"I don't know"。
## over-refusal的两种情况：

**Static conflict**   当在大语言模型的特征空间内的相似样本接收到不同的监督信号（原始的与修改后的 “我不知道”）时，就会发生静态冲突。 

**原因**：是两类问题在样本空间中的余弦相似度接近。文章中用  相似样本冲突率 $CRSS $来衡量**静态冲突率**，较高的CRSS表明存在更多冲突的相似样本对，可能导致过度拒绝。

$$
CRSS=\frac{\sum_{x \in D_{idk}}1(max_{x_j \in D_{van}}cos(r_i,r_j)>\tau_{sim})}{|D_{idk}|}
$$

**解决方法**：文章只说了采用Cor-Cer-RAIT之后$CRSS$ 减小，得知静态冲突变小。

**Dynamic conflict**  在监督微调（SFT）过程中，语言模型（LLM）的知识不断演化，从而能够回答之前无法回答的问题。然而，这些现在能够回答的训练样本仍然保留着基于初始语言模型状态的 “我不知道” 监督信号，进而导致了不一致的情况。 

**解决方法：**

​	对于**正确性**，去掉$`D_{idk}`$中训练前后模型回答问题中 **正确性提高**的部分

​	对于**确定性**，去掉$`D_{idk}`$中高确定性问题和$D_{van}$中**低确定性**的部分

**感知流** 就是 **正确性** 和 **确定性** 的变化

**正确性：**

$$
Cor(\mathcal{M},x_i)=\frac{1}{N}\sum_{\hat{a_j}\in \hat{A}_{i}}1(\hat{a_j}=x_i.a)
$$

**确定性：**

$$
Cer(\mathcal{M},x_i)=\frac{1}{N(N-1)}\sum_{\hat{a_j},\hat{a_k}\in \hat{A}_{i},j\ne k}\cos(E(a_j),E(a_k))
$$

其中 $`\hat{A_i}`$ 是对于问题$i$的多次回答答案的集合， $`E(a_j)`$ 是回答 $`a_j`$ 的表征向量。

## 方法：

##### Stage 1: Query the Knowledge State and Flow of LLM  	

首先，执行知识状态查询，以获得模型对源数据集中样本的响应的正确性和确定性。然后，对模型进行预演训练，从而得到被扰动的版本。通过比较扰动前后的知识状态，我们推导出在有监督的微调过程中知识流的指标。

##### Stage 2: Refusal-Aware Instructions Construction and Tuning  	

使用第一阶段的知识状态和流程，我们从 $D_{src}$ 中选择合适的样品来构建 RAIT数据，用于微调初始模型。