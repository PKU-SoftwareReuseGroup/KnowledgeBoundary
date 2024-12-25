from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from zhipuai import ZhipuAI


# 国内 huggingface 镜像
HF_ENDPOINT = 'https://hf-mirror.com'
# 自定义模型和分词器的下载路径
HF_HOME = '/data/data_public/ysq/models'
# 我购买的 glm-4-air 的key
GLM_KEY = "d9cbdfde73f23d6f1161dddd61ac92b4.Lhl52rvE0GCX2kDK"
client = ZhipuAI(api_key=GLM_KEY)


def get_TOKENIZER_and_MODEL(model_name: str):
    """根据 model_name 返回模型和分词器
    中文模型的分词器基于 `tiktoken` https://hf-mirror.com/Qwen/Qwen-7B#tokenizer
    Args: model_name: 模型名称
    Returns: tokenizer, model
    """
    tokenizer, model = None, None

    if model_name == "openlm-research/open_llama_3b":
        tokenizer = AutoTokenizer.from_pretrained(
            "openlm-research/open_llama_3b", use_fast=True, unk_token="<unk>",bos_token="<s>",eos_token="</s>",add_bos_token=False,
            cache_dir=HF_HOME,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "openlm-research/open_llama_3b", device_map='auto',
            cache_dir=HF_HOME,
        )

    elif model_name == "THUDM/chatglm2-6b":
        # https://hf-mirror.com/THUDM/chatglm2-6b
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

    elif model_name == "Qwen/Qwen-7B":
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

    # NOTE 目前效果最好的模型
    # 原模型是 5-shots，微调过后是 0-shot
    elif model_name == "Qwen/Qwen2-1.5B-Instruct":
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2-1.5B-Instruct",
            cache_dir=HF_HOME
        )
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2-1.5B-Instruct",
            torch_dtype="auto",
            device_map="auto",
            cache_dir=HF_HOME
        )

    elif model_name == "Qwen/Qwen2.5-3B":
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
    
    else:
        raise Exception("模型尚未支持")
    
    return tokenizer, model