
import json
import random
import math
import os
import numpy as np

#CORCER 路径
pre_res_path = "/data/data_public/breeze/KnowledgeBoundary/1_Qwen2.5-3B-Instruct_MMLU/evalution_res/01_origin/MMLU_CORCER.json"
aft_res_path = "/data/data_public/breeze/KnowledgeBoundary/1_Qwen2.5-3B-Instruct_MMLU/evalution_res/finetuned/CORCER_finetuned.json"
#RAIT数据集 路径
RAITset_path = "/data/data_public/breeze/KnowledgeBoundary/1_Qwen2.5-3B-Instruct_MMLU/ModelGenData/01_origin/MMLU_LMFlow.json"

#结果路径
res_COR_Change_Data_path = "./RES_DATASET/sub_CorChange.json"
res_COR_Change_Path = "./RES_DATASET/COR_Change.json"
res_CER_Sort_Path = "./RES_DATASET/CER_Sort.json"
res_CER_Change_Path = "./RES_DATASET/CER_Select.json"
res_CER_Change_Data_path = "./RES_DATASET/QWen2.5-3B/sub_CorCerChange.json"
res_CER_Change_Data_path_LF = "./RES_DATASET/QWen2.5-3B/sub_CorCerChange_LF.json"

#根据CER筛选数据集的比例，保留度
RATIO = 0.5

#为了LLaMa-Factory 进行微调，修改数据集格式
PATH_LF_RAIT = f"/data/data_public/breeze/KnowledgeBoundary/1_Qwen2.5-3B-Instruct_MMLU/ModelGenData/01_origin/MMLU_LF.json"
PATHRES_LF_RAIT = f"./RES_DATASET/QWen2.5-3B_finetined_LF/MMLU_LF_{RATIO}.json"


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

os.makedirs(os.path.dirname(res_CER_Change_Data_path), exist_ok=True) 
with open(res_CER_Change_Data_path,'w') as f:
    json.dump(RAIT, f)






with open(PATH_LF_RAIT,'r') as f:
    origin_lf = json.load(f)

newres = []
for dict_lf in origin_lf:
    for dict in new_instances:
        if dict_lf['input'] in dict['text']:
            newres.append(dict_lf)

os.makedirs(os.path.dirname(PATHRES_LF_RAIT), exist_ok=True) 
with open(PATHRES_LF_RAIT,'w') as f:
    json.dump(newres, f)