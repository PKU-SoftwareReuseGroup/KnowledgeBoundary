from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch

HF_HOME = '/data/data_public/ysq/models'

prompt="""
The following are multiple choice questions (with answers) about abstract algebra.

Find all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.
A. 0
B. 1
C. 2
D. 3
Answer: B.
Reasons: [...]

Now, please analyze the following question. Firstly, choose the correct answer from options A, B, C, and D. Then give reasons for your judgment.

Statement 1 | Every solvable group is of prime-power order. Statement 2 | Every group of prime-power order is solvable.
A. True, True
B. False, False
C. True, False
D. False, True
Answer:
"""

Q = "What is the native language of Jean Giraudoux?"
QA_input = f"""Now I have a Question: {Q}
Please answer the question based on your internal knowledge:
"""

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    cache_dir=HF_HOME
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    cache_dir=HF_HOME
)

# tokenizer = AutoTokenizer.from_pretrained(
#     "Qwen/Qwen2-1.5B-Instruct",
#     cache_dir=HF_HOME
# )
# model = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen2-1.5B-Instruct",
#     torch_dtype="auto",
#     device_map="auto",
#     cache_dir=HF_HOME
# )
# messages = [
#     {"role": "system", "content": f"You are a Knowledge Q&A expert."},
#     {"role": "user", "content": QA_input},
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=32,
#     # FIXME 显式指定 pad_token 避免控制台显示 Setting pad_token_id to eos_token_id:151643 for open-end generation
#     pad_token_id=0
# )
# generated_ids = [ output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids) ]
# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# print(response.split('.')[0] + ".")

# tokenizer = AutoTokenizer.from_pretrained(
#     "Qwen/Qwen2.5-3B",
#     cache_dir=HF_HOME
# )
# model = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen2.5-3B",
#     torch_dtype="auto",
#     device_map="auto",
#     cache_dir=HF_HOME
# )
# messages = [
#     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#     {"role": "user", "content": prompt},
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
# )
# # print(text)
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=512,
# )
# generated_ids = [
#     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(response)



# tokenizer = AutoTokenizer.from_pretrained(
#     # Qwen-7B 不支持添加 unknown special tokens
#     "Qwen/Qwen-7B", use_fast=True,
#     trust_remote_code=True, 
#     cache_dir=HF_HOME
# )
# model = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen-7B", device_map='auto',
#     trust_remote_code=True,
#     cache_dir=HF_HOME
# ).eval()
# inputs = tokenizer(prompt, return_tensors="pt").to(0)
# # print(inputs.keys())
# length = len(inputs['input_ids'][0])
# outputs = model.generate(
#         # 这里展开了 'input_ids' 'attention_mask' 等信息
#         **inputs,
#         max_new_tokens = 1,
#         output_scores = True,   # 输出模型生成的每个 token 的分数（logits）
#         return_dict_in_generate=True    # 以字典形式返回生成的输出，其中包括生成的 tokens 和 logits
#     )
# print(tokenizer.decode(outputs['sequences'][0][length:]))
# logits = outputs['scores'][0][0]
# print(logits[tokenizer("A").input_ids[0]])
# print(logits[tokenizer("B").input_ids[0]])
# print(logits[tokenizer("C").input_ids[0]])
# print(logits[tokenizer("D").input_ids[0]])
# probs = (
#     torch.nn.functional.softmax(
#         # 对 logits 进行 softmax 转换，将 logits 转化为概率分布，
#         # dim=0 表示在第一个维度上应用 softmax，即对所有选项的 logits 进行归一化
#         torch.tensor(
#             [
#                 # tokenizer 将每个选项 转化为对应的 token ID
#                 # 通过索引获取给定选项（A、B、C、D）对应的 logits 分数。
#                 logits[tokenizer("A").input_ids[0]],
#                 logits[tokenizer("B").input_ids[0]],
#                 logits[tokenizer("C").input_ids[0]],
#                 logits[tokenizer("D").input_ids[0]],
#             ],
#             dtype=torch.float32
#         ),
#         dim=0,
#     )
#     # 将 PyTorch 的 tensor 转换为 NumPy 数组，并确保不会有梯度计算（detach），并将其从 GPU 移动到 CPU 上
#     .detach().cpu().numpy()
# )


# tokenizer = AutoTokenizer.from_pretrained(
#     "THUDM/chatglm2-6b",
#     trust_remote_code=True,
#     # padding_side="left",
#     cache_dir=HF_HOME
# )
# model = AutoModel.from_pretrained(
#     "THUDM/chatglm2-6b",
#     trust_remote_code=True,
#     device='cuda',
#     cache_dir=HF_HOME
# ).eval()
# response, history = model.chat(tokenizer, prompt, history=[])
# print(response)


# tokenizer = AutoTokenizer.from_pretrained(
#     "openlm-research/open_llama_3b", use_fast=True, unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False,
#     cache_dir=HF_HOME,
# )
# model = AutoModelForCausalLM.from_pretrained(
#     "openlm-research/open_llama_3b", device_map='auto',
#     cache_dir=HF_HOME,
# )

# inputs = tokenizer(prompt, return_tensors="pt").to(0)
# ids = inputs['input_ids']
# length = len(ids[0])
# outputs = model.generate(
#         # 这里展开了 'input_ids' 'attention_mask' 等信息
#         ids,
#         max_new_tokens = 1,
#         output_scores = True,   # 输出模型生成的每个 token 的分数（logits）
#         return_dict_in_generate=True    # 以字典形式返回生成的输出，其中包括生成的 tokens 和 logits
#     )
# # print(tokenizer.decode(outputs[0]))
# # print(outputs.keys())
# # print(outputs['sequences'])
# print(tokenizer.decode(outputs['sequences'][0][length:]))
# logits = outputs['scores'][0][0]
# print(logits[tokenizer("A").input_ids[0]])
# print(logits[tokenizer("B").input_ids[0]])
# print(logits[tokenizer("C").input_ids[0]])
# print(logits[tokenizer("D").input_ids[0]])