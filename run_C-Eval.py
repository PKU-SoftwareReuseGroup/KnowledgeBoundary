import json
import os
import random
from argparse import ArgumentParser
from sys import exit

import numpy as np
import torch
from tqdm.auto import tqdm
# 引入自定义函数
from constants import get_TOKENIZER_and_MODEL

choices = ["A", "B", "C", "D"]


def format_example(input_list):
    prompt = input_list[0]
    k = len(input_list) - 2
    for j in range(k):
        prompt += f"\n{choices[j]}. {input_list[j+1]}"
    prompt += "\nAnswer:"
    return prompt


def format_shots(prompt_data):
    prompt = ""
    for data in prompt_data:
        prompt += data[0]       # fewshot 的 Question
        k = len(data) - 3       # fewshot 中作为选项的 Answer 的个数 NOTE C-Eval 有 explanation 所以-3
        for j in range(k):      # 格式化加入 A B C D 四个选项
            prompt += f"\n{choices[j]}. {data[j+1]}"
        prompt += "\nAnswer:"
        prompt += data[k+1] + "\n\n"

    return prompt


def gen_prompt(input_list: list[str], subject:str, prompt_data: list[list[str]]):
    """ 基于 C-Eval 数据生成 prompt
    这个 prompt 是针对 “如果模型还没有被调成一个 chatbot” 的情况。所以不应该加入 指令性语句
    而是以 fewshots 的形式让模型进行 text completion/genertion
    Params:
        input_text: C-Eval 的多选题数据 Question, Option1, Option2, Option3, Option4, Answer
        subject: C-Eval 的 key——领域名称
        prompt_data: 5-shots.json 在 subject 领域下的内容
    """
    # NOTE 领域介绍
    # prompt = f"The following are multiple choice questions (with answers) about{subject}.\n\n"
    prompt = ""
    # NOTE fewshot 构建
    prompt += format_shots(prompt_data)
    # NOTE 问题加入
    prompt += format_example(input_list)
    return prompt


def inference(tokenizer, model, full_input, subject):
    
    if args.model == "Qwen/Qwen2-1.5B-Instruct" or args.model == "Qwen/Qwen2.5-3B-Instruct":
        messages = [
            {"role": "system", "content": f"You are an expert on {subject}. Just give your answer between A, B, C, D, don't say anything else."},
            # {"role": "system", "content": f"你是 {subject} 领域的专家. 你需要对用户给出的最后一个问题选择一个答案。请确保你一定先回答选项。"},
            {"role": "user", "content": full_input}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=10,
            temperature=0.1,
            # FIXME 显式指定 pad_token 避免控制台显示 Setting pad_token_id to eos_token_id:151643 for open-end generation
            # pad_token_id=0,
            output_scores= True,
            return_dict_in_generate=True
        )
        # print(outputs.keys()) # dict_keys(['sequences', 'scores', 'past_key_values'])
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs['sequences'])
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(response)

        # logits 为模型输出第 1 个 token 的各种可能的 raw 预测分数
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
        return output_text, probs, response

    raise NotImplementedError(f"Model {args.model} not supported for inference.")


if __name__ == "__main__":
    parser = ArgumentParser()
    # 数据集，这里默认是 训练集
    parser.add_argument('--dataset', type=str, default="train")
    parser.add_argument('--prompt', type=str, default="5-shots")
    parser.add_argument('--model', type=str, required=True)

    args = parser.parse_args()
    
    tokenizer, model = get_TOKENIZER_and_MODEL(args.model)

    # 用于 LMFlow 的微调数据
    training_data = []
    LMFlow_data = {"type":"text_only","instances":[]}
    # 用于 Llama Factory 的微调数据
    LlamaFactory_data = []
    # 用于测试模型知识边界的数据
    data = []
    # 用于测试模型知识边界的 fewshot 数据
    prompt = []

    # 读取数据KnowledgeBoundary/data/C-Eval/5-shots.json
    with open(f"./data/C-Eval/{args.dataset}.json",'r') as f:
        data = json.load(f)
    
    with open(f"./data/C-Eval/{args.prompt}.json",'r') as f:
        prompt = json.load(f)

    print(f"🐟C-Eval datasets num of domains: {len(data)}")

    # 统计通过率
    Calcu_PASS = {}
    TOTAL, PASS = 0, 0
    # 统计每一个问题的 “正确性”Cor 和 “确定性”Cer
    CORCER={}
    texttmp = ""
    anstmp = ""

    for domain in tqdm(data.keys()):
        # if domain != "operating_system":
        #     continue
        # 分领域统计
        Calcu_PASS[domain] = {
            "PASS": 0,
            "TOTAL": 0,
            "ACC": 0.0000
        }
        CORCER[domain] = {}

        for sample in tqdm(data[domain]):
            # 初始化问题的“正确性”和“确定性”
            CORCER[domain][sample[0]] = {
                "COR": 0.0000,
                "CER": 0.0000
            }

            full_input = gen_prompt(sample, domain, prompt[domain])
            output, probs, _ = inference(tokenizer, model, full_input, domain)
            
            text = full_input
            texttmp = format_example(sample)
            # 如果模型输出的答案在标准答案中，则认为回答正确
            if sample[5] in output:
                anstmp = sample[5]
                text += f"{sample[5]}."
                Calcu_PASS[domain]["PASS"] += 1
                PASS += 1
                # 统计问题的“正确性”：模型给出 正确回答的概率
                # NOTE 这里其实就进行了一部分CorCer-RAIT Figure 4(c) 对于左上角 [D1_drop] 的删除，如果模型给出正确答案的概率不是最高，我们记为 0，默认它低于阈值 τ
                # 这样试图避免 错误集 经过微调 进入正确集，产生动态冲突
                CORCER[domain][sample[0]]["COR"] = probs[np.argmax(probs)].astype(float)
            # 否则认为回答错误。
            # 回答错误，即为不确定unsure，我们希望训练模型拒绝回答，用 N 表示
            else:
                text += "N." 
                anstmp = "N"

            training_data.append({"text": text})
                
            LlamaFactory_data.append({
                "instruction": "Output as N means the knowledge you are not sure about,and output as one of A, B, C, D means the knowledge you are certain about.",
                "input": texttmp,
                "output": anstmp
            })

            # 统计问题的“确定性”，注意将 float32 转化为 float，不然 JSON 不支持
            # 当存在 0 时 转换为非常小的数字，避免 log(0) 无穷大
            np_probs = np.array(probs)
            np_probs = np.where(np_probs == 0, 1e-9, np_probs)
            log_probs = np.log(np_probs)
            # 计算交叉熵
            CORCER[domain][sample[0]]["CER"] = -np.sum(np_probs * log_probs).astype(float)

            if np.isnan(CORCER[domain][sample[0]]["CER"]) or np.isnan(CORCER[domain][sample[0]]["COR"]):
                print(f"⚠ Error during inference: {CORCER[domain][sample[0]]}\n发生错误的问题是{sample[0]}\n")
                print(_)
                print(np_probs)
                print(log_probs)
                exit(0)
            
            # 统计领域问题数 和 总问题数
            Calcu_PASS[domain]["TOTAL"] += 1
            TOTAL += 1

        # 计算领域通过率 
        Calcu_PASS[domain]["ACC"] = round(Calcu_PASS[domain]["PASS"] / Calcu_PASS[domain]["TOTAL"], 4)

    # exit(0)

    model_name = f"{args.model}".split('/')[-1]

    # 导出 LMFlow 的微调数据
    random.shuffle(training_data)
    LMFlow_data['instances'] = training_data

    os.makedirs("./training_data", exist_ok=True)
    os.makedirs(f"./training_data/{model_name}", exist_ok=True)
    with open(f"./training_data/{model_name}/C-Eval_LMFlow.json",'w') as f:
        json.dump(LMFlow_data, f)
    # 导出 Llama Factory 的微调数据
    with open(f"./training_data/{model_name}/C-Eval_LF.json",'w') as f:
        json.dump(LlamaFactory_data, f)

    # 导出模型通过率统计结果【知识边界】
    Calcu_PASS["Final_Evaluation"] = {
        "Pass": PASS,
        "Total": TOTAL,
        "Accuarcy": round(PASS/TOTAL, 4)
    }
    os.makedirs("./2.1_evalution_res", exist_ok=True)
    os.makedirs(f"./2.1_evalution_res/{model_name}", exist_ok=True)
    with open(f"./2.1_evalution_res/{model_name}/C-Eval_Pass.json", "w") as f:
        json.dump(Calcu_PASS, f)

    # 导出问题的“正确性”和“确定性”统计结果
    with open(f"./2.1_evalution_res/{model_name}/C-Eval_CORCER.json", "w") as f:
        json.dump(CORCER, f)