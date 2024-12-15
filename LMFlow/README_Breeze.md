## 操作流程

先下载llama3b 模型

```
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download openlm-research/open_llama_3b --local-dir ../../models/openlm-research/open_llama_3b
```

注意修改路径

```python
python run_llama3b_MMLU.py #用MMLU数据集对llama3b模型进行提问并统计ACC，同时计算每一个问题的COR和CER

./scripts/run_finetune.sh #将产生的问答结果用来微调模型 

python run_llama3b_finetune_MMLU.py #用MMLU数据集对微调后的模型进行提问并统计ACC，同时计算每一个问题的COR和CER

python run_select_dataset.py #删掉训练前后的Dink中变成正确的部分，结果存放于RES_DATASET中

python run_calc.py #统计 筛选数据集 前后 正确个数、正确个数提升、正确个数下降、正确率提升、确定性提升等

```