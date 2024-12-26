from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
from scipy.stats import entropy
import math
import os
import numpy as np

DATASET= "/data/data_public/breeze/KnowledgeBoundary/data/MMLU/MMLU_ID_train.json"
MODELPATH = "/data/data_public/breeze/finetuned_models/Qwen2_1.5B-Instruct-LFfinefune_0.2"

choices = ["A", "B", "C", "D"]

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
    # prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
    #     format_subject(subject)
    # )
    
    # prompt+= "If you know what the answer is, just print the answer, not the content of the answer.If you're not sure about the generated answer or you don't know how to answer the question, output: N."
    
    # prompt += """
    # If you know what the answer is, just print the answer, not the content of the answer.
    #         "Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.",
    #         "0",
    #         "1",
    #         "2",
    #         "3",
    #         "B"
    # If you're not sure about the generated answer or you don't know how to answer the question, output: N.
    #             "Statement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.",
    #         "True, True",
    #         "False, False",
    #         "True, False",
    #         "False, True",
    #         "N"
    # """
    
    #prompt += format_shots(prompt_data)
    
    prompt = format_example(input_list)
    
    return prompt

def inference(tokenizer,model,input_text,subject,prompt_data):
    
    full_input = gen_prompt(input_text,subject,prompt_data)
    
    messages = [
        # {"role": "system", "content": f"If you know what the answer is, just print the answer, not the content of the answer.If you're not sure about the generated answer or you don't know how to answer the question, output: N."},
        {"role": "system", "content": f"You are an expert on {subject}."},
        {"role": "user", "content": full_input}
    ]

    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(0)
    
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=1,
        output_scores= True,
        return_dict_in_generate=True
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs['sequences'])
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    logits = outputs['scores'][0][0]

    
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
    
    #output_text = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
    output_text = {0: "A", 1: "B", 2: "C", 3: "D", 4: "N"}[np.argmax(probs)]
    # if output_text == "N":
    #     output_text = "I don't know"
    
    # print("----------------------------------------")
    # print(output_text)
    # print("----------------------------------------")
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
    
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    LMFlow_data = {"type":"text_only","instances":[]}
    factorydata = []

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
    TOTAL_UNPASS,TOTAL_REFUSE = 0,0

    #统计每一个问题的Cor和Cer
    CORCER={}
    texttmp = ""
    anstmp = ""
    #data 是训练数据集
    #sample[0]是问题，sample[5]是问题的标准答案
    datalist = list(data.keys())
    for domain in tqdm(datalist): 
        #分领域统计
        Calcu_PASS[domain] = {
            "PASS":0,
            "UNPASS":0,
            "REFUSE":0,
            "TOTAL":0,
            "ACC":0.0000 
        }
        CORCER[domain] = {}
        
        for sample in tqdm(data[domain]):
            
            CORCER[domain][sample[0]]={
                "COR":0.0000,
                "CER":0.0000
            }
            
            if args.method == "unsure":
                output, full_input, probs= inference(tokenizer,model,sample,domain,prompt[domain])

                text = f"{full_input}"

                texttmp = format_example(sample)

                if sample[5] in output:
                    anstmp = sample[5]
                    text += f"{sample[5]}."
                    Calcu_PASS[domain]["PASS"]+=1
                    TOTAL_PASS+=1
                    CORCER[domain][sample[0]]["COR"] = probs[np.argmax(probs)].astype(float) #统计回答正确的问题的COR
                else:
                    if 'N' in output:
                        anstmp = "N"
                        TOTAL_REFUSE += 1
                        Calcu_PASS[domain]["REFUSE"]+=1
                    else:
                        anstmp = "Error"
                        TOTAL_UNPASS+=1
                        Calcu_PASS[domain]["UNPASS"]+=1
                        text += f"Error:{output}."
                
                factorydata.append({
                    "instruction": "Output as N means the knowledge you are not sure about,and output as one of A, B, C, D means the knowledge you are certain about.",
                    "input": texttmp,
                    "output": anstmp
                })
                #统计问题的精确度,注意将float32转化为float,不然JSON不支持
                #当0时 转换为非常小的数字
                np_probs = np.array(probs)
                np_probs = np.where(np_probs == 0, 1e-9, np_probs)
                log_probs = np.log(np_probs)
                
                CORCER[domain][sample[0]]["CER"]=-np.sum(np_probs*log_probs).astype(float)
                
                training_data.append({"text":text})
                Calcu_PASS[domain]["TOTAL"] += 1
                TOTAL+=1
            Calcu_PASS[domain]["ACC"] = round(Calcu_PASS[domain]["PASS"]/Calcu_PASS[domain]["TOTAL"], 4)
    random.shuffle(training_data)
    LMFlow_data['instances'] = training_data

    modelname = f"{args.model}".split('/')[-1]
    os.makedirs(f"./ModelGenData/{modelname}",exist_ok=True)
    with open(f"./ModelGenData/{modelname}/{args.result}.json",'w') as f:
        json.dump(LMFlow_data,f)

    with open(f"./ModelGenData/{modelname}/{args.result}_LF.json",'w') as f:
        json.dump(factorydata,f)
    
    Calcu_PASS["Final_Evaluation"] = {
        "Pass": TOTAL_PASS,
        "UNPASS":TOTAL_UNPASS,
        "REFUSE":TOTAL_REFUSE,
        "Total": TOTAL,
        "Accuarcy": round(TOTAL_PASS/TOTAL, 4)
    }
    os.makedirs(f"./evalution_res/{modelname}", exist_ok=True)
    with open(f"./evalution_res/{modelname}/{modelname}.json", "w") as f:
        json.dump(Calcu_PASS, f)
        
    with open(f"./evalution_res/{modelname}/CORCER_{modelname}.json", "w") as f:
        json.dump(CORCER, f)
