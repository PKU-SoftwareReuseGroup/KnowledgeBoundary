# 参考 R-tuning 项目 https://github.com/shizhediao/R-Tuning/
# 中文模型的分词器基于tiktoken https://hf-mirror.com/Qwen/Qwen-7B#tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
from scipy.stats import entropy
import math
import os
import numpy as np
# 引入自定义变量
from constants import HF_HOME
from sys import exit

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


def gen_prompt(input_list: list[str], subject:str, prompt_data: list[list[str]]):
    """ 基于 MMLU 数据生成 prompt
    Params:
        input_text: MMLU 的多选题数据 Question, Option1, Option2, Option3, Option4, Answer
        subject: MMLU 的 key——领域名称
        prompt_data: prompt.json 在 subject 领域下的内容
    """
    # NOTE 领域介绍
    prompt = f"The following are multiple choice questions (with answers) about{subject}.\n\n"
    # NOTE fewshot 构建
    for data in prompt_data:
        # fewshot 的 Question
        prompt += data[0]
        # fewshot 中作为选项的 Answer 的个数
        k = len(data) - 2
        # 格式化加入 A B C D 四个回答
        for j in range(k):
            prompt += f"\n{choices[j]}. {data[j+1]}"
        prompt += f"\nAnswer:{data[k+1]}\n\n"
    # NOTE 模型求解问题构建，我这里加入了指导性语句
    # prompt += "Now, please analyze the following question, choose the correct answer from options A, B, C, and D. and give reasons for your judgment.\n"
    prompt += input_list[0]
    k = len(input_list) - 2
    for j in range(k):
        prompt += f"\n{choices[j]}. {input_list[j+1]}"
    prompt += "\nAnswer:"
    return prompt

# Qwen2.5-3B 用完整的 5-shot prompt 无法生成结果，且回答总是先理由再选项。用 1-shot + 指令规范
def gen_one_shot_prompt(input_list: list[str], subject:str, data: list[str]):
    # NOTE 领域介绍
    prompt = f"The following are multiple choice questions (with answers) about{subject}.\n\n"
    # NOTE oneshot 构建，这里固定先回答选项，再回答其他
    prompt += data[0]
    k = len(data) - 2
    for j in range(k):
        prompt += f"\n{choices[j]}. {data[j+1]}"
    prompt += f"\nAnswer: {data[k+1]}.\n\n"
    # NOTE 模型求解问题构建，我这里加入了指导性语句
    prompt += "Now, there is a question for you: \n"
    prompt += input_list[0]
    k = len(input_list) - 2
    for j in range(k):
        prompt += f"\n{choices[j]}. {input_list[j+1]}"

    prompt += "\nYour task: Firstly, choose the correct answer from options A, B, C, and D. Secondly, give reasons.\n"
    return prompt


def inference(
        tokenizer, 
        model, 
        full_input: str, 
        subject:str
    ):
    """
    Params:
        tokenizer: 分词器
        model: 用于推理的模型
        full_input: 5-shots prompt 或者 1-shot prompt
        subject: 领域名称
    """
    output_text = ""
    if args.model == "openlm-research/open_llama_3b":
        inputs = tokenizer(full_input,return_tensors="pt").to(0)
        ids = inputs['input_ids']
        length = len(ids[0])
        # 输出有三个键值 ['sequences', 'scores', 'past_key_values']
        outputs = model.generate(
                ids,
                # NOTE 这里限制新生成的 token 长度为1，其实并不能保证生成一个字母，可能是其他字符
                max_new_tokens = 20,
                output_scores = True,   # 输出模型生成的每个 token 的分数（logits）
                return_dict_in_generate=True    # 以字典形式返回生成的输出，其中包括生成的 tokens 和 logits
            )
        # print(tokenizer.decode(outputs['sequences'][0][length:]))
        # logits 是模型输出的对每个可能的 token 的原始分数，通常是一个向量，表示所有词汇表中每个 token 的 "raw" 预测得分
        logits = outputs['scores'][0][0]    #The first token
        # print(logits)
        # print(logits[tokenizer("A").input_ids[0]])
        # probs 是一个包含四个选项 A、B、C、D 对应的概率分布。包含四个浮点数的数组
        probs = (
            torch.nn.functional.softmax(
                # 对 logits 进行 softmax 转换，将 logits 转化为概率分布，
                # dim=0 表示在第一个维度上应用 softmax，即对所有选项的 logits 进行归一化
                torch.tensor(
                    [
                        # tokenizer 将每个选项 转化为对应的 token ID
                        # 通过索引获取给定选项（A、B、C、D）对应的 logits 分数。
                        logits[tokenizer("A").input_ids[0]],
                        logits[tokenizer("B").input_ids[0]],
                        logits[tokenizer("C").input_ids[0]],
                        logits[tokenizer("D").input_ids[0]],
                    ]
                ),
                dim=0,
            )
            # 将 PyTorch 的 tensor 转换为 NumPy 数组，并确保不会有梯度计算（detach），并将其从 GPU 移动到 CPU 上
            .detach().cpu().numpy()
        )
        # print(probs)
        # 取概率最大的作为索引，映射到字母作为 推理输出
        output_text = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

    elif args.model == "THUDM/chatglm2-6b":
        response, history = model.chat(tokenizer, full_input, history=[])
        # chatglm2-6b 的回复一般是 [选项]'.' [选项内容]. [理由]
        output_text = response.split('.')[0]

    elif args.model == "Qwen/Qwen-7B":
        inputs = tokenizer(full_input,return_tensors="pt").to(0)
        # print(inputs.keys())
        outputs = model.generate(
                # 这里展开了 'input_ids' 'attention_mask' 等信息
                **inputs,
                max_new_tokens = 1,
                output_scores = True,   # 输出模型生成的每个 token 的分数（logits）
                return_dict_in_generate=True    # 以字典形式返回生成的输出，其中包括生成的 tokens 和 logits
            )
        # print(outputs.keys())
        # print(outputs['sequences'])
        # print(tokenizer.decode(outputs['sequences'][0][length:]))
        logits = outputs['scores'][0][0]
        # print(logits[tokenizer("A").input_ids[0]])
        probs = (
            torch.nn.functional.softmax(
                # 对 logits 进行 softmax 转换，将 logits 转化为概率分布，
                # dim=0 表示在第一个维度上应用 softmax，即对所有选项的 logits 进行归一化
                torch.tensor(
                    [
                        # tokenizer 将每个选项 转化为对应的 token ID
                        # 通过索引获取给定选项（A、B、C、D）对应的 logits 分数。
                        logits[tokenizer("A").input_ids[0]],
                        logits[tokenizer("B").input_ids[0]],
                        logits[tokenizer("C").input_ids[0]],
                        logits[tokenizer("D").input_ids[0]],
                    ],
                    # NOTE Qwen 的向量格式是 bf16，需要转换
                    dtype=torch.float32
                ),
                dim=0,
            )
            # 将 PyTorch 的 tensor 转换为 NumPy 数组，并确保不会有梯度计算（detach），并将其从 GPU 移动到 CPU 上
            .detach().cpu().numpy()
        )
        # print(probs)
        # 取概率最大的作为索引，映射到字母作为 推理输出
        output_text = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
    
    elif args.model == "Qwen/Qwen2.5-3B":
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        # FIXME 注意这里 Qwen2.5-3B 用的 prompt 和其他模型不一致
        messages = [
            {"role": "system", "content": f"You are an expert on{s}. You must answer me first and then give me your reasons."},
            {"role": "user", "content": full_input},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            # FIXME 这里显式指定 pad_token 只是为了不要让控制台显示 Setting pad_token_id to eos_token_id:151643 for open-end generation 这条信息
            pad_token_id=0
        )
        generated_ids = [ output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids) ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(response)
        tmpList = response.split('.')
        output_text = f"{tmpList[0]}. {tmpList[1]}"
        # print(output_text)
    return output_text


if __name__ == "__main__":
    print(os.getcwd())
    
    parser = ArgumentParser()
    # NOTE 训练集默认使用 MMLU_ID_train.json
    parser.add_argument('--dataset', type=str, default="MMLU_ID_train")
    # NOTE 初始提示符默认使用 MMLU_ID_prompt.json
    parser.add_argument('--prompt_domain', type=str, default="ID",choices=["ID","OOD"])
    parser.add_argument('--model', type=str, required=True)
    # parser.add_argument('--result',type=str, default="MMLU")
    parser.add_argument('--method',type=str,default="unsure",choices=["unsure","unknown","uncertain"])
    parser.add_argument("--num_try",type=int,default="5") #only required for uncertain method
    
    args = parser.parse_args()
    
    if args.model == "openlm-research/open_llama_3b":
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, use_fast=True, unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False,
            cache_dir=HF_HOME,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map='auto',
            cache_dir=HF_HOME,
        )
    
    elif args.model == "THUDM/chatglm2-6b": # https://hf-mirror.com/THUDM/chatglm2-6b
        tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/chatglm2-6b",
            trust_remote_code=True,
            # padding_side="left",
            cache_dir=HF_HOME
        )
        model = AutoModel.from_pretrained(
            "THUDM/chatglm2-6b",
            trust_remote_code=True,
            device='cuda',
            cache_dir=HF_HOME
        ).eval()

    elif args.model == "Qwen/Qwen-7B":
        # https://hf-mirror.com/Qwen/Qwen-7B 上描述它在 MMLU 上的5-shot表现是 58.2
        # 需要安装 tiktoken transformers_stream_generator
        # 警告安装 flash-attn https://github.com/Dao-AILab/flash-attention
        tokenizer = AutoTokenizer.from_pretrained(
            # Qwen-7B 不支持添加 unknown special tokens
            "Qwen/Qwen-7B", use_fast=True,
            trust_remote_code=True, 
            cache_dir=HF_HOME
        )
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-7B", device_map='auto',
            trust_remote_code=True,
            cache_dir=HF_HOME
        ).eval()

    elif args.model == "Qwen/Qwen2.5-3B":
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-3B",
            cache_dir=HF_HOME
        )
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B",
            torch_dtype="auto",
            device_map="auto",
            cache_dir=HF_HOME
        )

    elif args.model == "Qwen/Qwen2-1.5B": # https://hf-mirror.com/Qwen/Qwen2-1.5B
        pass
    else:
        raise Exception("模型尚未支持")

    # print(f"================模型设备检查: {model.device}================")
    # exit(0)

    LMFlow_data = {"type":"text_only","instances":[]}

    training_data = []
    uncertain_data = []
    data = []
    prompt = []
    uncertain_data = []
    with open(f"./data/MMLU/{args.dataset}.json",'r') as f:
        data = json.load(f)
    
    with open(f"./data/MMLU/MMLU_{args.prompt_domain}_prompt.json",'r') as f:
        prompt = json.load(f)
    
    # 这里存放知识边界评估的结果 knowledge_boundary_eval
    KB_eval = {}
    MMLU_pass, MMLU_total = 0, 0
    # NOTE 遍历 MMLU 的各个领域（MMLU 是一个字典）
    for i in tqdm(data.keys()):
        # if i != "college_mathematics":
        #     continue
        KB_eval[i] = {
            "Pass": 0,
            "Total": 0,
            "Accuarcy": 0.0000
        }
        # NOTE 各个子领域的 value 是一个 list[list[str]]
        # sample 是 list[str] 的多选题数据：Question, Option1, Option2, Option3, Option4, Answer
        for sample in tqdm(data[i]):
        # for sample in tqdm(data[i][1:4]):
            # NOTE 把 领域名称 中的下划线替换为空格，并在开头加一个空格
            l = i.split("_")
            subject = ""
            for entry in l:
                subject += " " + entry
            # NOTE 构建 模型输入文本
            full_input = ""
            if args.model != "Qwen/Qwen2.5-3B":
                full_input = gen_prompt(sample, subject, prompt[i])
            else:
                # 选择 第一个 例子 作为 one_shot
                full_input = gen_one_shot_prompt(sample, subject, prompt[i][0])

            # NOTE 利用 R-Tuning 论文的 padding 方式评估模型边界，生成训练数据集
            if args.method == "unsure":
                output = inference(tokenizer, model, full_input, i)
                
                text = f"{full_input}{sample[5]}. Are you sure you accurately answered the question based on your internal knowledge?"
                if sample[5] in output:
                    text += " I am sure."
                    KB_eval[i]["Pass"]+=1
                    MMLU_pass+=1
                else:
                    text += " I am unsure."  
                
                training_data.append({"text":text})
                KB_eval[i]["Total"]+=1
                MMLU_total+=1
            else:
                raise Exception("不支持的方法")
            
        KB_eval[i]["Accuarcy"] = round(KB_eval[i]["Pass"]/KB_eval[i]["Total"], 4)
        print(KB_eval[i])

    random.shuffle(training_data)
    LMFlow_data['instances'] = training_data

    model_name = args.model.split("/")[1]

    os.makedirs("./training_data", exist_ok=True)
    os.makedirs(f"./training_data/{model_name}", exist_ok=True)
    with open(f"./training_data/{model_name}/MMLU_{args.method}.json",'w') as f:
        json.dump(LMFlow_data,f)

    KB_eval["Final_Evaluation"] = {
        "Pass": MMLU_pass,
        "Total": MMLU_total,
        "Accuarcy": round(MMLU_pass/MMLU_total, 4)
    }
    os.makedirs("./2.1_evalution_res", exist_ok=True)
    with open(f"./2.1_evalution_res/KB_for_{model_name}_on_MMLU.json", "w") as f:
        json.dump(KB_eval, f)
