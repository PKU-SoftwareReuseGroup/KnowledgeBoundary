# 宋春风修正
run_finetune.sh 更改路径
修正,路径是文件夹，不是文件
```python
model_name_or_path=/data/data_public/ysq/models/models--openlm-research--open_llama_3b/snapshots/141067009124b9c0aea62c76b3eb952174864057
dataset_path=/data/data_public/ysq/KnowledgeBoundary/training_data/open_llama_3b
output_dir=output_models/finetuned_llama_3b 
```
运行,出现错误
```python
 NotImplementedError: Using RTX 4000 series doesn't support faster communication broadband via P2P or IB. Please set `NCCL_P2P_DISABLE="1"` and `NCCL_IB_DISABLE="1" or use `accelerate launch` which will do this automatically.
```
RTX 4000 系列显卡在设计上可能没有对 P2P（Peer - to - Peer，对等网络）和 IB（InfiniBand，一种高速网络通信技术）这种高速通信带宽提供支持。
解决方法：在运行 Python 脚本之前手动设置环境变量来禁用这些通信模式，用命令行设置环境变量：
```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```
出现错误:
```bash
deepspeed.ops.op_builder.builder.CUDAMismatchException: >- DeepSpeed Op Builder: Installed CUDA version 11.8 does not match the version torch was compiled with 12.4, unable to compile cuda/cpp extensions without a matching cuda version.
```
服务器cuda与torch版本对不上，卸载torch后重装cuda11.8的torch相关模块
```bash
pip uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
出现错误：
```bash
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: ERROR api_key not configured (no-tty). call wandb.login(key=[your_api_key])
```
表示该框架需要wandb可视化,申请wandb账号并获得自己的APIKEY
```python
export WANDB_API_KEY = < your W&B KEY >
```

因版本原因（该错误不影响）,在 src/lmflow/pipeline/finetuner.py中，第577行：用processing_class替换tokenizer
```python
#tokenizer=model.get_tokenizer(),
processing_class = model.get_tokenizer(),
```

