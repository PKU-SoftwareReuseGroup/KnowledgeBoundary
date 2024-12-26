## Task 2.2.2 避免知道的知识被错回复为“不知道”，降低或者避免over-refusal

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

&emsp;&emsp;用MMLU数据集询问原模型，前加5个 `fewshot`， `message`设置如下：

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

&emsp;&emsp;用MMLU数据集询问原模型，基本不给任何提示信息, message设置如下：

```python
    messages = [
        {"role": "system", "content": f"You are an expert on {subject}."},
        {"role": "user", "content": full_input}
    ]
```

但是对于模型生成结果设置中：

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

对于模型 `qwen2-1.5B-Instruct` 模型：

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

![Origin Model](C:\Users\79378\Desktop\assets\Origin Model-1735110141085-10.png)

![First finetuned Model](C:\Users\79378\Desktop\assets\First finetuned Model-1735110149336-13.png)

![The dataset was filtered with a threshold of 0.5](C:\Users\79378\Desktop\assets\The dataset was filtered with a threshold of 0.5-1735110197352-17.png)

![The dataset was filtered with a threshold of 0.2](C:\Users\79378\Desktop\assets\The dataset was filtered with a threshold of 0.2-1735110202232-19-1735110204164-21.png)

![Total Eva](C:\Users\79378\Desktop\assets\Total Eva.png)

