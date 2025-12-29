#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author : yp

import os
import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from liger_kernel.transformers import apply_liger_kernel_to_gemma2 

os.environ["SWANLAB_DISABLED"] = "1"
os.environ["SWANLAB_PROJECT"] = "Gemma-3-Alignment"

# 路径配置
TRAIN_PATHS = [
    "/workspace/training/datas/generation_data/hjb_dpo_std_format_sample123456.jsonl",
    "/workspace/training/datas/generation_data/synthesize_合成数据_hjb_from_query_100_dpo_format.jsonl",
    # "/workspace/training/datas/generation_data/train_ragtruth_dpo.jsonl",
    ]
# model_name = "/workspace/training/pretrained_model/gemma-3-12b-it-sp-end"
# model_name = "/workspace/training/models/gemma-3-12b-it-nobca-v1202-bf16/checkpoint-922"
model_name = "/workspace/training/models/gemma-3-12b-it-nobca-hjb-v1216-bf16/checkpoint-1230"
output_dir = "/workspace/training/models/gemma3_12b_it_sp_end_dpo_v1229_sft-bs-dpo-sam123456-query100"


def make_prompt_from_messages(messages):
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "").strip()
        if role == "system":
            parts.append(f"[System]: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    return "\n".join(parts)


def prepare_dataset(paths):
    ds = load_dataset("json", data_files=paths, split="train")
    def build_example(ex):
        # 注意：这里确保 chosen 和 rejected 是纯文本
        query = make_prompt_from_messages(ex.get("messages", []))
        chosen = ex.get("chosen", "")
        rejected = ex.get("rejected", "")
        return {"prompt": query, "chosen": chosen, "rejected": rejected}

    # 移除原始列，只保留 DPO 需要的列
    ds = ds.map(build_example, remove_columns=ds.column_names)
    return ds

# 1. 配置参数
torch_dtype = torch.bfloat16
attn_implementation = "flash_attention_2"

# 2. 加载数据
dataset = prepare_dataset(TRAIN_PATHS)

def check_data(ex):
    return len(ex["chosen"]) > 5 and len(ex["rejected"]) > 5

dataset = dataset.filter(check_data)
dataset = dataset.shuffle(seed=42)
data_split = dataset.train_test_split(test_size=0.05, seed=42)

train_dataset = data_split["train"]
eval_dataset = data_split["test"]

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(eval_dataset)}")

dataset_len = len(train_dataset)

# === 1. 获取训练超参数 ===
per_device_batch_size = 1        # 你代码里的设置
gradient_accumulation_steps = 4  # 你代码里的设置
num_gpus = torch.cuda.device_count() # 自动获取显卡数量
if num_gpus == 0: num_gpus = 1   # 防止 CPU 调试报错

# === 2. 计算 Global Batch Size 和 Steps per Epoch ===
global_batch_size = per_device_batch_size * gradient_accumulation_steps * num_gpus
steps_per_epoch = math.ceil(dataset_len / global_batch_size)

print(f"--- 动态步数计算 ---")
print(f"数据量: {dataset_len}")
print(f"Global Batch Size: {global_batch_size}")
print(f"每个 Epoch 的步数: {steps_per_epoch}")

# === 3. 动态策略设置 ===
eval_steps = min(500, max(10, int(steps_per_epoch * 0.1))) # 最多500步验证一次
save_steps = min(2000, max(50, int(steps_per_epoch * 0.5))) # 最多2000步保存一次

print(f"动态设置 -> Eval steps: {eval_steps}, Save steps: {save_steps}")
print(f"------------------")

# 3. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.config.use_cache = False

print("正在加载 Reference Model (ZeRO-3 Requirement)...")
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
)
ref_model.config.use_cache = False
# Reference Model 不需要计算梯度，设置 eval 模式
ref_model.eval()


import numpy as np # 需要安装 numpy: pip install numpy

def print_dataset_statistics(dataset, tokenizer):
    """
    统计 DPO 数据集的 Token 长度分布
    """
    print("\n" + "="*20 + " 数据集长度统计开始 " + "="*20)
    
    # 定义处理函数，计算每条数据的长度
    def compute_length(example):
        # 你的 prepare_dataset 已经生成了纯文本的 prompt, chosen, rejected
        # 这里不需要加 special tokens，因为 DPO Trainer 内部处理时会加
        # 但为了保险起见，统计时通常可以假设会有少量 overhead，直接计算即可
        
        p_ids = tokenizer.encode(example["prompt"], add_special_tokens=False)
        c_ids = tokenizer.encode(example["chosen"], add_special_tokens=False)
        r_ids = tokenizer.encode(example["rejected"], add_special_tokens=False)
        
        len_p = len(p_ids)
        len_c = len(c_ids)
        len_r = len(r_ids)
        
        return {
            "len_prompt": len_p,
            "len_chosen": len_c,
            "len_rejected": len_r,
            "len_total_chosen": len_p + len_c,
            "len_total_rejected": len_p + len_r
        }

    # 多进程处理以加速 (根据你的 CPU 核数调整 num_proc)
    # 假设 dataset 比较大，先取 train_dataset 进行统计
    print("正在计算 Token 长度 (这可能需要几秒钟)...")
    stats_dataset = dataset.map(compute_length, num_proc=16, remove_columns=dataset.column_names)
    
    # 转换为 numpy 数组方便计算
    len_prompts = np.array(stats_dataset["len_prompt"])
    len_total_chosen = np.array(stats_dataset["len_total_chosen"])
    len_total_rejected = np.array(stats_dataset["len_total_rejected"])
    
    # 辅助打印函数
    def print_percentiles(name, data):
        p50 = np.percentile(data, 50)
        p90 = np.percentile(data, 90)
        p95 = np.percentile(data, 95)
        p99 = np.percentile(data, 99)
        p_max = np.max(data)
        print(f"[{name}]")
        print(f"  Avg: {np.mean(data):.1f}")
        print(f"  P50: {int(p50)}")
        print(f"  P90: {int(p90)}")
        print(f"  P95: {int(p95)} <--- 建议参考这个值")
        print(f"  P99: {int(p99)}")
        print(f"  Max: {int(p_max)}")
        print("-" * 30)
        return int(p95), int(p99)

    # 打印统计信息
    p95_prompt, _ = print_percentiles("Prompt Length (提示词长度)", len_prompts)
    p95_total_c, _ = print_percentiles("Total Length (Prompt + Chosen)", len_total_chosen)
    p95_total_r, _ = print_percentiles("Total Length (Prompt + Rejected)", len_total_rejected)
    
    # 给出建议
    suggested_max_len = max(p95_total_c, p95_total_r) + 64 # 留一点 buffer
    suggested_prompt_len = p95_prompt + 32
    
    print(f"\n>>> 推荐设置建议 (基于 P95 覆盖率):")
    print(f"max_prompt_length ~= {suggested_prompt_len}")
    print(f"max_length        ~= {suggested_max_len}")
    print("="*60 + "\n")

# 4. 加载 Tokenizer (关键修复部分)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
print_dataset_statistics(train_dataset, tokenizer)

# Gemma/Llama 通常没有 pad_token，直接将其指向 eos_token_id 即可
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
ref_model.config.pad_token_id = tokenizer.pad_token_id # ref_model 也要设置

# 确保 pad_token_id 是 int 类型
assert isinstance(tokenizer.pad_token_id, int), "Pad token ID must be an integer"

# 显式设置 model config，防止警告
model.config.pad_token_id = tokenizer.pad_token_id
apply_liger_kernel_to_gemma2(model)


from peft import LoraConfig

peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)


# 5. 配置 DPO
dpo_config = DPOConfig(
    output_dir=output_dir,
    beta=0.1,
    loss_type="sigmoid",
    bf16=True,
    tf32=True,
    per_device_train_batch_size=per_device_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    learning_rate=5e-7,
    warmup_ratio=0.1,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="steps",
    save_steps=save_steps,
    eval_strategy="steps",
    eval_steps=eval_steps,
    max_length=6000,
    max_prompt_length=6000,
)


# 6. 初始化 Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

# === 新增：打印训练参数数量 ===
# 只有主进程打印，避免多卡刷屏
if trainer.accelerator.is_main_process:
    print("\n" + "="*20 + " LoRA 参数统计 " + "="*20)
    # 获取被封装后的 PEFT 模型
    # DPOTrainer 内部会将 model 包装成 PeftModel
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    else:
        # 备用统计方法
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in trainer.model.parameters())
        print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}%")
    print("="*60 + "\n")

print("Starting training...")
trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# bash
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --config_file seqparallel_config.yaml --main_process_port 29501 train_dpo.py

# === yaml
# compute_environment: LOCAL_MACHINE
# debug: false
# distributed_type: DEEPSPEED
# downcast_bf16: 'no'
# machine_rank: 0
# main_training_function: main
# mixed_precision: bf16
# num_machines: 1
# num_processes: 6
# rdzv_backend: static
# same_network: true
# use_cpu: false
# deepspeed_config:
#   # 对应你代码中的 gradient_accumulation_steps
#   gradient_accumulation_steps: 4
#   gradient_clipping: 1.0
  
#   # 【核心优化】ZeRO Stage 3：切分参数、梯度和优化器状态
#   zero_stage: 3
  
#   # 【防爆显存】将优化器状态卸载到 CPU 内存
#   offload_optimizer_device: cpu
#   offload_optimizer_nvme_path: null
  
#   # 【防爆显存】将参数卸载到 CPU 内存
#   offload_param_device: cpu
#   offload_param_nvme_path: null
  
#   # 【初始化优化】在加载模型时就进行切分，防止瞬间 OOM
#   zero3_init_flag: true
  
#   # 保存时合并权重，方便后续使用
#   zero3_save_16bit_model: true
  
#   # 混合精度配置
#   bf16:
#     enabled: true
