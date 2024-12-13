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
