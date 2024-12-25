import torch
import json
from tqdm.auto import tqdm
import random
from argparse import ArgumentParser
from scipy.stats import entropy
import math
import os

# 引入自定义变量
from constants import client, get_TOKENIZER_and_MODEL
# 引入 exit 用于调试
from sys import exit


def gen_prompt(input_text: str):
    if args.model == "openlm-research/open_llama_3b":
        return "Question:" + input_text + " Answer:"
    # 其他模型用比较强的提示
    return f"Now I have a Question: {input_text}\nPlease answer the question based on your internal knowledge: "


def inference(full_input: str):
    """
    Params:
        tokenizer: 分词器
        model: 用于推理的模型
        full_input: 0-shot prompt 开放域知识问答没有例子
    """
    output_text = ""
    if args.model == "openlm-research/open_llama_3b":
        inputs = tokenizer(full_input,return_tensors="pt").to(0)
        ids = inputs['input_ids']
        length = len(ids[0])
        
        outputs = model.generate(
                ids,
                #temperature=0.7,
                #do_sample = True,
                max_new_tokens = 15,
            )
        # FIXME 因为这个模型是一个补全模型，生成的内容是接续在输入之后，所以要从这以后 截取输出
        output_text = tokenizer.decode(outputs[0][length:])
        # print(f"数据库问题\n{full_input}")
        # print(f"模型对数据库问题完整的回答\n{output_text}")
        idx = output_text.find('.')
        output_text = output_text[:idx]

    elif args.model == "THUDM/chatglm2-6b":
        response, history = model.chat(tokenizer, full_input, history=[])
        # print(f"response: {response}")
        output_text = response.split('.')[0] + "."

    elif args.model == "Qwen/Qwen2.5-3B" or args.model == "Qwen/Qwen2-1.5B-Instruct":
        messages = [
            {"role": "system", "content": f"You are a Knowledge Q&A expert."},
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
            max_new_tokens=32,
            # FIXME 显式指定 pad_token 避免控制台显示 Setting pad_token_id to eos_token_id:151643 for open-end generation
            pad_token_id=0
        )
        generated_ids = [ output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids) ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(response)
        output_text = response.split('.')[0] + "."

    return output_text


def judge_answer_similarity(q: str, a1: str, a2: str):
    full_input=f"""问题: {q}
回答1: {a1}
回答2: {a2}
说明: 回答1 是问题的标准答案。回答2 可能会给出一些多余的信息。
如果 回答2 中的信息和 回答1 含义相似（不区分大小写），或者从另一个方面回答了和 回答1 的相似的含义，那么我们认为它是合格的。
如果 回答2 对于 问题 的回答 和 回答1 有较大的差异，或者 回答2 给出的答案过度宽泛，那么我们认为它是不合格的。
注意: 回答2 是否是一个完整的句子，是否包含了其他信息，**不作为其是否合格的判断依据**
任务: 判断回答2的正确性，回答 YES 或者 NO，并另起一行 **简要** 给出判断理由。

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
            {"role": "system", "content": "你是一个评判者，你只会基于用户提供的信息，不利用你自己掌握的知识。"},
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
    
    tokenizer, model = get_TOKENIZER_and_MODEL(args.model)

    LMFlow_data = {"type":"text_only","instances":[]}

    training_data = []
    data = []
    with open(f"./data/pararel/{args.dataset}.json",'r') as f:
        data = json.load(f)
    
    print(f"🐟ParaRel datasets length: {len(data)}")


    # 这里存放知识边界评估的结果 knowledge_boundary_eval
    ParaRel_pass, ParaRel_total = 0, 0
    # NOTE 只在评估知识边界时，统计经过 GLM-4-Air 二次评价后 模型的准确率
    #  对于 2.2.1 任务，还是从模型忠诚度的角度出发，用 in 的方式来严格限制
    Judge_pass = 0

    # NOTE sample[0] 是问题. sample[1] 是回答. 这里似乎抛弃了 sample[2]
    # for sample in tqdm(data[100:115]):
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
                # if sample[1] in output or sample[1].capitalize() in output:
                if sample[1].lower() in output.lower():
                    text += " I am sure."
                    ParaRel_pass += 1
                    Judge_pass += 1
                else:
                    text += " I am unsure."
                    judge_res = judge_answer_similarity(sample[0], sample[1], output)
                    if "YES" in judge_res:
                        # text += " I am sure."
                        # ParaRel_pass += 1
                        Judge_pass += 1
                    # else:
                    #     text += " I am unsure."
            except Exception as e:
                print(f"⚠ Error during inference: {e}\n发生错误的是{sample[0]}\n{sample[1]}\n")
                # 如果遇到敏感内容错误，直接认为 unsure
                text += " I am unsure."
                
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
        "JudgePass": Judge_pass,
        "Total": ParaRel_total,
        "Accuarcy": round(ParaRel_pass/ParaRel_total, 4),
        "JudgeAccuarcy": round(Judge_pass/ParaRel_total, 4)
    }
    os.makedirs("./2.1_evalution_res", exist_ok=True)
    os.makedirs(f"./2.1_evalution_res/{model_name}", exist_ok=True)
    with open(f"./2.1_evalution_res/{model_name}/KB_on_ParaRel.json", "w") as f:
        json.dump(KB_eval, f)

