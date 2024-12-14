
import json
from tqdm.auto import tqdm
import random
import math
import os
import numpy as np

pre_res_path = "./2.2.2_1_evalution_res/CORCER_open_llama_3b.json"
aft_res_path = "./2.2.2_1_evalution_res/CORCER_finetuned_llama_3b.json"

RAITset_path = "./2.2.2_0_ModelGenData/finetuned_llama_3b/MMLU_unsure.json"
res_set_path = "./RES_DATASET/sub_corretChange.json"

with open(pre_res_path,'r') as f:
    origin = json.load(f)

with open(aft_res_path,'r') as f:
    finetune = json.load(f)
    
cor_Change = {}
for domain_Q in origin.keys():
    # print(domain_Q)
    for question in origin[domain_Q]:
        if(origin[domain_Q][question]["COR"]==0.0 and finetune[domain_Q][question]["COR"] > 0): #从错误变为正确
            if domain_Q not in cor_Change:
                cor_Change[domain_Q]=[]
                cor_Change[domain_Q].append(question)
            else:
                cor_Change[domain_Q].append(question)

with open(RAITset_path,'r') as f:
    RAIT = json.load(f)

questionlist =[]
for domain in cor_Change.keys():
    questionlist+=list(cor_Change[domain])

#在RAIT中删除掉Dink又变正确的数据集
new_instances = []
for instance in RAIT["instances"]:
    new_instance = {}
    for ele in instance:
        match = False
        for question in questionlist:
            if(question ==  instance[ele].split("\n\n")[-1].split('\n')[0]):
                match = True
                break                
        if not match:
            new_instance[ele] = instance[ele]
        if new_instance:
            new_instances.append(new_instance)

RAIT["instances"]=new_instances
               
os.makedirs(res_set_path.split('/')[-2], exist_ok=True)
with open(res_set_path,'w') as f:
    json.dump(RAIT, f)