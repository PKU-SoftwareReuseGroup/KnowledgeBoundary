from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
from scipy.stats import entropy
import math
import os
import numpy as np

DATASET= "../data/MMLU/MMLU_ID_train.json"
MODELPATH = "../../models/openlm-research/open_llama_3b"

choices = ["A", "B", "C", "D"]

FALSE_RESPONSES = ["The answer is unknown.",
                   "The answer is uncertain.",
                   "The answer is unclear.",
                   "It is not known.",
                   "I do not know the answer.",
                   "I'm not sure.",
                   "There is no definitive answer.",
                   "There is much debate.",
                   "There is no concrete answer to this question.",
                   "It is impossible to answer.",
                   "There is no known case.",
                   "There is no public information available.",
                   "There is no scientific evidence.",
                   "There is no right answer.",
                   "It is impossible to know.",
                   "It is difficult to predict.",
                   ]

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(input_list):
    prompt = input_list[0]
    k = len(input_list) - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], input_list[j+1])
    prompt += "\nAnswer:"
    return prompt

def format_shots(prompt_data):
    prompt = ""
    for data in prompt_data:
        prompt += data[0]
        k = len(data) - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], data[j+1])
        prompt += "\nAnswer:"
        prompt += data[k+1] + "\n\n"

    return prompt


def gen_prompt(input_list,subject,prompt_data):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    prompt += format_shots(prompt_data)
    prompt += format_example(input_list)
    return prompt

def inference(tokenizer,model,input_text,subject,prompt_data):
    full_input = gen_prompt(input_text,subject,prompt_data)
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    length = len(ids[0])
    if args.method == "uncertain":
        outputs = model.generate(
                ids,
                temperature=0.7,
                do_sample = True,
                max_new_tokens = 1,
            )
        output_text = tokenizer.decode(outputs[0][length:])
    else:       
        outputs = model.generate(
                ids,
                max_new_tokens = 1,
                output_scores = True,
                return_dict_in_generate=True
            )
        logits = outputs['scores'][0][0]
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
        output_text = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
    return output_text, full_input,probs

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default=DATASET)
    parser.add_argument('--prompt_domain', type=str, default="ID",choices=["ID","OOD"])
    parser.add_argument('--model', type=str, default=MODELPATH)
    parser.add_argument('--result',type=str, default="MMLU")
    parser.add_argument('--method',type=str,default="unsure",choices=["unsure","unknown","uncertain"])
    parser.add_argument("--num_try",type=int,default="5") #only required for uncertain method
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=True,unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False)
    model = AutoModelForCausalLM.from_pretrained(args.model,device_map='auto')

    LMFlow_data = {"type":"text_only","instances":[]}

    training_data = []
    uncertain_data = []
    data = []
    prompt = []
    uncertain_data = []
    with open(f"{args.dataset}",'r') as f:
        data = json.load(f)
    
    with open(f"../data/MMLU/MMLU_{args.prompt_domain}_prompt.json",'r') as f:
        prompt = json.load(f)
        
    # 统计通过率
    Calcu_PASS = {}
    TOTAL,TOTAL_PASS = 0,0
    #统计每一个问题的Cor和Cer
    CORCER={}
    #data 是训练数据集
    #sample[0]是问题，sample[5]是问题的标准答案
    datalist = list(data.keys())
    for i in tqdm(datalist): 
        #分领域统计
        Calcu_PASS[i] = {
            "PASS":0,
            "TOTAL":0,
            "ACC":0.0000  
        }
        CORCER[i] = {}
        for sample in tqdm(data[i]):
            CORCER[i][sample[0]]={
                "COR":0.0000,
                "CER":0.0000
            }
            if args.method == "unsure":
                output, full_input, probs= inference(tokenizer,model,sample,i,prompt[i])
                text = f"{full_input}{sample[5]}. Are you sure you accurately answered the question based on your internal knowledge?"
                
                if sample[5] in output:
                    text += " I am sure."
                    Calcu_PASS[i]["PASS"]+=1
                    TOTAL_PASS+=1
                    CORCER[i][sample[0]]["COR"] = probs[np.argmax(probs)].astype(float) #统计回答正确的问题的COR
                else:
                    text += " I am unsure." 
                #统计问题的精确度,注意将float32转化为float,不然JSON不支持
                np_probs = np.array(probs)
                log_probs = np.log(np_probs)
                CORCER[i][sample[0]]["CER"]=-np.sum(np_probs*log_probs).astype(float)
                training_data.append({"text":text})
                Calcu_PASS[i]["TOTAL"] += 1
                TOTAL+=1
            elif args.method == "unknown":
                output, full_input = inference(tokenizer,model,sample,i,prompt[i])
                if sample[5] in output:
                    text = f"{full_input}{sample[5]}."
                else:
                    random_int = random.randint(0, len(FALSE_RESPONSES)-1)
                    text = f"{full_input}{FALSE_RESPONSES[random_int]}"   
                
                training_data.append({"text":text})
            
            elif args.method == "uncertain":
                answers = []
                occurance = {}
                for j in range(args.num_try):
                    output, full_input = inference(tokenizer,model,sample,i,prompt[i])
                    answers.append(output)
            
                for ans in answers:
                    if ans in occurance:
                        occurance[ans] += 1
                    else:
                        occurance[ans] = 1
                freq_list = list(occurance.values())
                answer_entropy = entropy(freq_list)
                uncertain_data.append((answer_entropy,f"{full_input}{sample[5]}."))
            Calcu_PASS[i]["ACC"] = round(Calcu_PASS[i]["PASS"]/Calcu_PASS[i]["TOTAL"], 4)
    if args.method == "uncertain":
        uncertain_data.sort(key=lambda x: x[0])
        split_half = math.floor(len(uncertain_data)*0.5)
        for (answer_entropy,sample) in uncertain_data[:split_half]:
            text = f"{sample} Are you sure you accurately answered the question based on your internal knowledge?"
            training_data.append({"text":f"{text} I am sure."})
            
        for (answer_entropy,sample) in uncertain_data[split_half:]:
            text = f"{sample} Are you sure you accurately answered the question based on your internal knowledge?"
            training_data.append({"text":f"{text} I am unsure."})

    random.shuffle(training_data)
    LMFlow_data['instances'] = training_data

    modelname = f"{args.model}".split('/')[-1]
    os.makedirs(f"./2.2.2_0_ModelGenData/{modelname}",exist_ok=True)
    with open(f"./2.2.2_0_ModelGenData/{modelname}/{args.result}_{args.method}.json",'w') as f:
        json.dump(LMFlow_data,f)
    
    Calcu_PASS["Final_Evaluation"] = {
        "Pass": TOTAL_PASS,
        "Total": TOTAL,
        "Accuarcy": round(TOTAL_PASS/TOTAL, 4)
    }
    os.makedirs("./2.2.2_1_evalution_res", exist_ok=True)
    with open(f"./2.2.2_1_evalution_res/{modelname}.json", "w") as f:
        json.dump(Calcu_PASS, f)
        
    with open(f"./2.2.2_1_evalution_res/CORCER_{modelname}.json", "w") as f:
        json.dump(CORCER, f)
