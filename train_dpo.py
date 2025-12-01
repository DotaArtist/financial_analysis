#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --config_file fsdp_config.yaml train_dpo.py

# Author : yp

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

# 关闭 swanlab (如果需要)
os.environ["SWANLAB_DISABLED"] = "1"

# 路径配置
TRAIN_PATHS = ["/workspace/training/datas/generation_data/anthropic_hhrlhf_train.jsonl"]
model_name = "/workspace/training/pretrained_model/gemma-3-12b-it-sp-end"
output_dir = "/workspace/training/models/dpo_gemma3_12b_it_sp_end"

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

# 3. 加载模型
# 关键修改：不要手动 .to("cuda")，交给 Accelerator/Trainer 自动管理设备
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.config.use_cache = False

# 4. 加载 Tokenizer (关键修复部分)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# 【核心修复】：不要 resize embeddings，直接复用 eos 作为 pad
# Gemma/Llama 通常没有 pad_token，直接将其指向 eos_token_id 即可
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 确保 pad_token_id 是 int 类型
assert isinstance(tokenizer.pad_token_id, int), "Pad token ID must be an integer"

# 显式设置 model config，防止警告
model.config.pad_token_id = tokenizer.pad_token_id

# 5. 配置 DPO
dpo_config = DPOConfig(
    output_dir=output_dir,
    beta=0.05,
    loss_type="ipo",
    bf16=True,
    tf32=True,
    per_device_train_batch_size=1, # H200 显存大，建议设为 4 或 8
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=5e-7,
    warmup_ratio=0.1,
    num_train_epochs=1,
    logging_steps=10,
    save_strategy="steps",
    save_steps=1000,
    max_length=2048,
    max_prompt_length=1024,
)



# 6. 初始化 Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer, # 新版 TRL 建议用 processing_class 替代 tokenizer 参数
)

print("Starting training...")
trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir) # 别忘了保存 tokenizer
