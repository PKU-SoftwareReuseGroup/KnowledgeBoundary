import json
import random
import math
import os
import numpy as np

PATH_ORIGIN = "./2.2.2_1_evalution_res/CORCER_open_llama_3b.json"
PATH_FINETUNE = "./2.2.2_1_evalution_res/CORCER_finetuned_llama_3b.json"
PATH_FINETUNE_2 = "./2.2.2_1_evalution_res/CORCER_finetuned_2_llama_3b.json"

PATH_RESULT = "./2.2.2_1_evalution_res/SUM_CORCER_change.json"

with open(PATH_ORIGIN,'r') as f:
    origin = json.load(f)
    
with open(PATH_FINETUNE,'r') as f:
    finetune = json.load(f)

with open(PATH_FINETUNE_2,'r') as f:
    finetune_2 = json.load(f)

'''
result = {
    domain_1: #领域1
    {
        "TOTAL":n_TOTAL,
        "CORRECT":n_COR,
        "COR up": n_COR_up,
        "COR down": n_COR_down,
        "CER up": n_CER_up,
        "CER down": n_CER_down,
    },   
     
    ...
    
    domain_n: #领域n
    {
        "TOTAL":n_TOTAL,
        "CORRECT":n_COR,
        "COR up": n_COR_up,
        "COR down": n_COR_down,
        "CER up": n_CER_up,
        "CER down": n_CER_down,
    },  
    TOTALDATA:
        domain_n: #领域n
    {
        "TOTAL":n_TOTAL,
        "CORRECT":n_COR,
        "COR up": n_COR_up,
        "COR down": n_COR_down,
        "CER up": n_CER_up,
        "CER down": n_CER_down,
    }
}
'''

result = {}
TOTOL,COR,CORRATEUP,CORRATEDOWN,CERRATEUP,CERRATEDOWN = 0,0,0,0,0,0
CORUP,CORDOWN = 0,0

for domain in origin.keys():
    n_TOTAL,n_COR,n_COR_Rate_up,n_CER_Rate_up = 0,0,0,0
    n_COR_Rate_down,n_CER_Rate_down = 0,0
    n_COR_up,n_COR_down = 0,0
    
    result[domain] = {}
    for question in origin[domain].keys():
        n_TOTAL += 1
        TOTOL += 1
        if finetune_2[domain][question]["COR"]>0:
            n_COR += 1
            COR+=1
            if finetune[domain][question]["COR"] == 0.0 :
                n_COR_up += 1
                CORUP += 1
        elif finetune[domain][question]["COR"] > 0:
            n_COR_down += 1
            CORDOWN+=1
            
        if finetune_2[domain][question]["COR"] > finetune[domain][question]["COR"]:
            n_COR_Rate_up += 1
            CORRATEUP += 1
        elif finetune_2[domain][question]["COR"] < finetune[domain][question]["COR"]:
            n_COR_Rate_down += 1
            CORRATEDOWN += 1
        if finetune_2[domain][question]["CER"] > finetune[domain][question]["CER"]:
            n_CER_Rate_up += 1
            CERRATEUP +=1
        elif finetune_2[domain][question]["CER"] < finetune[domain][question]["CER"]:
            n_CER_Rate_down += 1
            CERRATEDOWN += 1
    result[domain] = {
        "TOTAL":n_TOTAL,                    #该领域问题总数
        "CORRECT":n_COR,                    #回答正确的个数
        "CORRECT UP":n_COR_up,              #finetune回答错误，筛选数据集后微调模型 回答正确
        "CORRECT DOWN":n_COR_down,          #finetune回答正确，筛选数据集后微调模型 回答错误
        "COR rate up": n_COR_Rate_up,       #筛选数据集后微调 前后问题答案*正确性*提高的个数
        "COR rate down": n_COR_Rate_down,   #筛选数据集后微调 前后问题答案*正确性*降低的个数
        "CER rate up": n_CER_Rate_up,       #筛选数据集后微调 前后问题答案*确定性*提高的个数
        "CER rate down": n_CER_Rate_down,   #筛选数据集后微调 前后问题答案*确定性*降低的个数
    }
result["TOTALDATA"] = {
    "TOTAL":TOTOL,
    "CORRECT":COR,
    "CORRECT UP":CORUP,
    "CORRECT DOWN":CORDOWN,
    "COR rate up": CORRATEUP,
    "COR rate down": CORRATEDOWN,
    "CER rate up": CERRATEUP,
    "CER rate down": CERRATEDOWN,
}

with open(PATH_RESULT,'w') as f:
    json.dump(result,f)
