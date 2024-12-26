# 原模型测试参数设置：
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

# 微调后的参数设置
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