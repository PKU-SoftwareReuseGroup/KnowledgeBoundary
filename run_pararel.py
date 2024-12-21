from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
from scipy.stats import entropy
import math
import os

# 引入自定义变量
from constants import HF_HOME, client


def gen_prompt(input_text: str):
    return "Question:" + input_text + " Answer:"


def inference(full_input: str):
    """
    Params:
        tokenizer: 分词器
        model: 用于推理的模型
        full_input: 0-shot prompt 开放域知识问答没有例子
    """
    inputs = tokenizer(full_input,return_tensors="pt").to(0)
    ids = inputs['input_ids']
    length = len(ids[0])
     
    outputs = model.generate(
            ids,
            #temperature=0.7,
            #do_sample = True,
            # FIXME 截断模型输出？
            max_new_tokens = 15,
        )
    # FIXME 因为这个模型是一个补全模型，生成的内容是接续在输入之后，所以要从这以后 截取输出
    output_text = tokenizer.decode(outputs[0][length:])
    # print(f"数据库问题\n{full_input}")
    # print(f"模型对数据库问题完整的回答\n{output_text}")
    idx = output_text.find('.')
    output_text = output_text[:idx]
    return output_text


def judge_answer_similarity(q: str, a1: str, a2: str):
    full_input=f"""问题: {q}
回答1: {a1}
回答2: {a2}
说明: 回答1 是问题的标准答案。回答2 可能会给出一些多余的信息。
如果 回答2 中的信息和 回答1 含义相似（不区分大小写），或者从另一个方面回答了和 回答1 的相似的含义，那么我们认为它是合格的。
如果 回答2 对于 问题 的回答 和 回答1 有较大的差异，或者 回答2 给出的答案过度宽泛，那么我们认为它是不合格的。
注意: 回答2 是否是一个完整的句子，是否包含了其他信息，**不作为其是否合格的判断依据**
任务: 判断回答2的正确性，回答我 YES 或者 NO，并另起一行给出你的理由。

下面是一些例子，请参考后完成任务。

### 例子1
问题: In what field does Anaxagoras work?
回答1: philosophy
回答2: Anaxagoras was a Greek philosopher who lived in the
参考判断: 
YES
回答2 提供了 Anaxagoras 是希腊哲学家的事实，从另一个角度回答了 Anaxagoras 在哪一个领域工作。因此，回答2 中的信息和 回答1 相似，所以判断合格。

### 例子2
问题: What field does Robert Bunsen work in?
回答1: chemistry
回答2: Bunsen is a German chemist who is best known for his inventio
参考判断: 
YES
回答2 提供了 Bunsen 是德国化学家的事实，从另一个角度回答了 Bunsen 在哪一个领域工作。尽管 回答2 并不完整，并且含有其他信息，但是它包含了和 回答1 相似的信息，所以判断合格。

### 例子3
问题: What field does Alan Turing work in?
回答1: logic
回答2: Alan Turing is a computer scientist
参考判断: 
NO
回答2 描述 Alan Turing 是在计算机科学领域工作，而计算机科学领域 和 逻辑领域不存在直接的从属关系，因此，回答2 和 回答1 有较大的差异，所以判断不合格。
(尽管从客观事实来看，Alan Turing 确实在两个领域都有贡献，但是我们只参照 问题 和 回答1)

### 例子4
问题: What field does Bruce Perens work in?
回答1: programmer
回答2: Bruce Perens works in Software
参考判断:
NO
回答2 虽然指出了 Bruce Perens 在软件领域工作，但并未明确指出他的具体职业是程序员，而是提供了一个有些过于宽泛的领域。因此，回答2 和 回答1 有较大的差异，所以判断不合格。
(尽管在软件领域工作的人 有较大的概率有编程经验，并且 Bruce Perens 确实是一个程序员，但是回答2 的回复过于宽泛)
"""
    response = client.chat.completions.create(
        model="glm-4-air",
        messages=[
            {"role": "system", "content": "你是一个评判者，你只会基于用户提供的信息，不利用或者补充你自己掌握的知识。"},
            {"role": "user", "content": full_input}
        ],
        # temperature=0.3
    )
    response = response.choices[0].message.content
    # print("------- 评判回答一致性的 prompt -------")
    # print(full_input)
    # print("------- 评判回答一致性的 结果 -------")
    # print(response)
    idx = response.find('\n')
    response = response[:idx]
    return response


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="training_data")
    parser.add_argument('--model', type=str, required=True)
    # parser.add_argument('--result',type=str, default="pararel")
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

    LMFlow_data = {"type":"text_only","instances":[]}

    training_data = []
    data = []
    with open(f"./data/pararel/{args.dataset}.json",'r') as f:
        data = json.load(f)
    
    print(f"🐟ParaRel datasets length: {len(data)}")


    # 这里存放知识边界评估的结果 knowledge_boundary_eval
    ParaRel_pass, ParaRel_total = 0, 0

    # NOTE sample[0] 是问题. sample[1] 是回答. 这里似乎抛弃了 sample[2]
    # for sample in tqdm(data[:5]):
    for sample in tqdm(data):

        full_input = gen_prompt(sample[0])
        
        if args.method == "unsure":
            output = inference(full_input)
            
            text = f"Question: {sample[0]} Answer: {sample[1]}. Are you sure you accurately answered the question based on your internal knowledge?"
            # print(f"Ground Truth Q: {sample[0]}")
            # print(f"Ground Truth A: {sample[1]}")
            # print(f"Model Answer: {output}")
            # NOTE 对比 标准回答 A 和 模型回答 A' 划分 不确定集D_0、确定集D_1
            # 这里加一个主要是因为 只因为首字母不同导致判断失误的太多了
            try:
                if sample[1] in output or sample[1].capitalize() in output:
                    text += " I am sure."
                    ParaRel_pass += 1
                else:
                    judge_res = judge_answer_similarity(sample[0], sample[1], output)
                    if "YES" in judge_res:
                        # print("评估合格")
                        text += " I am sure."
                        ParaRel_pass += 1
                    else:
                        text += " I am unsure."
            except Exception as e:
                print(f"⚠ Error during inference: {e}\n发生错误的是{sample[0]}\n{sample[1]}\n")
                # 如果遇到敏感内容错误，直接认为 unsure
                text += " I am unsure."
                
            # print(text)
            training_data.append({"text":text})
            ParaRel_total += 1

        else:
            raise Exception("不支持的方法")

    random.shuffle(training_data)
    LMFlow_data['instances'] = training_data

    model_name = args.model.split("/")[1]

    os.makedirs("./training_data",exist_ok=True)
    os.makedirs(f"./training_data/{model_name}", exist_ok=True)
    with open(f"./training_data/{model_name}/ParaRel_{args.method}.json",'w') as f:
        json.dump(LMFlow_data,f)

    
    KB_eval = {
        "Pass": ParaRel_pass,
        "Total": ParaRel_total,
        "Accuarcy": round(ParaRel_pass/ParaRel_total, 4)
    }
    os.makedirs("./2.1_evalution_res", exist_ok=True)
    with open(f"./2.1_evalution_res/KB_for_{model_name}_on_ParaRel.json", "w") as f:
        json.dump(KB_eval, f)

