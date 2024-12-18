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
[TASK2.2.2 理论](./LMFlow/Task2_2_2.md)
进行finetune时，为避免模块冲突，请用conda复制一份环境，并还原`transformers == 4.47.0`和`tokenizers==0.21.0`版本
其他修正方式请参照[finetune管理文档](./LMFlow/FinetuneChange.md)

先下载llama3b 模型

```
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download openlm-research/open_llama_3b --local-dir ../../models/openlm-research/open_llama_3b
```

注意在相应文件中 **修改model和dataset路径**

```python
python run_llama3b_MMLU.py #用MMLU数据集对llama3b模型进行提问并统计ACC，同时计算每一个问题的COR和CER

./scripts/run_finetune.sh #将产生的问答结果用来微调模型 

python run_llama3b_finetune_MMLU.py #用MMLU数据集对微调后的模型进行提问并统计ACC，同时计算每一个问题的COR和CER

python run_select_dataset.py #删掉训练前后的Dink中变成正确的部分，结果存放于RES_DATASET中

python run_calc.py #统计 筛选数据集 前后 正确个数、正确个数提升、正确个数下降、正确率提升、确定性提升等

```