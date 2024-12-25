from argparse import ArgumentParser
import json
import os

import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
# 引入自定义函数
from constants import get_TOKENIZER_and_MODEL

# 这里将 C-Eval 数据集的 val 集作为测试集，dev 集作为测试集，将其转为 JSON
DATA_DIR = "./data/C-Eval"

if __name__ == "__main__":
    parser = ArgumentParser()
    # 数据集，这里默认是 训练集
    parser.add_argument('--dataset', type=str, default="training_data")
    # parser.add_argument('--model', type=str, required=True)

    args = parser.parse_args()
    
    # tokenizer, model = get_TOKENIZER_and_MODEL(args.model)
    
    test_df=pd.read_csv(os.path.join(DATA_DIR,"test","computer_network_test.csv"))