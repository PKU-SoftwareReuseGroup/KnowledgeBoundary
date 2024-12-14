
import json
from tqdm.auto import tqdm
import random
import math
import os
import numpy as np

pre_res_path = "./2.2.2_1_evalution_res/CORCER_open_llama_3b.json"
aft_res_path = "./2.2.2_1_evalution_res/CORCER_finetuned_llama_3b.json"

RAITset_path = "./2.2.2_0_ModelGenData/open_llama_3b/MMLU_unsure.json"

res_COR_Change_Data_path = "./2.2.2_2_RES_DATASET/sub_CorChange.json"

res_COR_Change_Path = "./2.2.2_2_RES_DATASET/COR_Change.json"

res_CER_Sort_Path = "./2.2.2_2_RES_DATASET/CER_Sort.json"
res_CER_Change_Path = "./2.2.2_2_RES_DATASET/CER_Select.json"

res_CER_Change_Data_path = "./2.2.2_2_RES_DATASET/sub_CerChange.json"
#根据CER筛选数据集的比例
RATIO = 0.5

with open(pre_res_path,'r') as f:
    origin = json.load(f)

with open(aft_res_path,'r') as f:
    finetune = json.load(f)
    
cor_Change = {} #统计Cor由0变正的问题集

for domain_Q in origin.keys():
    for question in origin[domain_Q]:
        if(origin[domain_Q][question]["COR"]==0.0 and finetune[domain_Q][question]["COR"] > 0): #从错误变为正确
            if domain_Q not in cor_Change:
                cor_Change[domain_Q]=[]
                cor_Change[domain_Q].append(question)
            else:
                cor_Change[domain_Q].append(question)

os.makedirs(res_COR_Change_Path.split('/')[-2], exist_ok=True) 
with open(res_COR_Change_Path,'w') as f:
    json.dump(cor_Change, f)

with open(RAITset_path,'r') as f:
    RAIT = json.load(f)

questionlist =[]
for domain in cor_Change.keys():
    questionlist+=list(cor_Change[domain])

#在RAIT中删除掉Dink变成正确的数据集
new_instances = []
for instance in RAIT["instances"]:
    new_instance = {}
    for ele in instance:
        match = False
        for question in questionlist:
            if(question in instance[ele]):
                match = True
                break                
        if not match:
            new_instance[ele] = instance[ele]
        if new_instance:
            new_instances.append(new_instance)

RAIT["instances"]=new_instances
       
os.makedirs(res_COR_Change_Data_path.split('/')[-2], exist_ok=True)
with open(res_COR_Change_Data_path,'w') as f:
    json.dump(RAIT, f)
    
    
#根据CER对数据集进行筛选
Cer_Sort={} 
for domain in origin.keys():
    dic = dict(origin[domain])
    Cer_Sort[domain] = sorted(dic.items(), key=lambda x: x[1]["CER"]) #对domain_Q领域的进行CER排序
os.makedirs(res_CER_Sort_Path.split('/')[-2], exist_ok=True) 
with open(res_CER_Sort_Path,'w') as f:
    json.dump(Cer_Sort, f)

#统计IDK和VAN的个数，以便筛选
sum_idk_van = {}
for domain in Cer_Sort.keys():
    dic = dict(Cer_Sort[domain])
    count_idk = sum(1 for key, value in dic.items() if value["COR"] == 0.0)
    count_van = sum(1 for key, value in dic.items() if value["COR"] > 0.0)
    sum_idk_van[domain]={
        "IDK":count_idk,
        "VAN":count_van           
    }
# print(sum_idk_van)

#挑选出根据CER 保留的问题集
select_questions = {}
for domain in Cer_Sort.keys():
    select_questions[domain] = []
    dic = dict(Cer_Sort[domain])
    n_Idk,n_Van = 0,0
    idk_threshold = int(sum_idk_van[domain]["IDK"] * RATIO)
    van_threshold = int(sum_idk_van[domain]["VAN"] * (1-RATIO))
    for key, value in dic.items():
        if(value["COR"] == 0.0 and n_Idk<=idk_threshold):
            n_Idk+=1
            select_questions[domain].append(key)
        elif(value["COR"] > 0.0):
            n_Van+=1
            if n_Van >= van_threshold:
                select_questions[domain].append(key)
                
os.makedirs(res_CER_Change_Path.split('/')[-2], exist_ok=True) 
with open(res_CER_Change_Path,'w') as f:
    json.dump(select_questions, f)

#在RAIT中 根据 select_questions 选择数据集
questionlist =[]
for domain in select_questions.keys():
    questionlist+=list(select_questions[domain])
    

new_instances = []
for instance in RAIT["instances"]:
    new_instance = {}
    for ele in instance:
        match = False
        for question in questionlist:
            if(question in  instance[ele]):
                match = True
                break                
        if match:
            new_instance[ele] = instance[ele]
        if new_instance:
            new_instances.append(new_instance)  
                       
RAIT["instances"]=new_instances     

os.makedirs(res_CER_Change_Data_path.split('/')[-2], exist_ok=True) 
with open(res_CER_Change_Data_path,'w') as f:
    json.dump(RAIT, f)