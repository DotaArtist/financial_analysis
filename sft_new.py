#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author : yp


"""
# Epoch	层级比例 (Layer 1-5)	学习率
# 1	50% / 30% / 15% / 5% / 0%	5e-5
# 2	30% / 30% / 25% / 10% / 5%	4e-5
# 3	15% / 20% / 30% / 25% / 10%	3e-5
# 4	10% / 15% / 25% / 30% / 20%	2e-5
# 5	5% / 10% / 20% / 35% / 30%	1e-5
#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1,2,3 && accelerate launch --num_processes 4 /workspace/training/training/lora.py \
#   --model_name /workspace/training/models/gemma-3-12b-it-sp-end-bf16 \
#   --train_path /workspace/training/datas/generation_data/train_bca-knowledge-qa-single-turn-epoch-1-b7_en_20251028_960.jsonl \
#   --save_model /workspace/training/models/gemma-3-12b-it-rewrite-v1029-epoch1-bf16 \
#   --learning_rate 0.00005

# export CUDA_VISIBLE_DEVICES=0,1,2,3 && accelerate launch --num_processes 4 /workspace/training/training/lora.py \
#   --model_name /workspace/training/models/gemma-3-12b-it-rewrite-v1029-epoch1-bf16 \
#   --train_path /workspace/training/datas/generation_data/train_bca-knowledge-qa-single-turn-epoch-2-b7_en_20251028_960.jsonl \
#   --save_model /workspace/training/models/gemma-3-12b-it-rewrite-v1029-epoch2-bf16 \
#   --learning_rate 0.00004

# export CUDA_VISIBLE_DEVICES=0,1,2,3 && accelerate launch --num_processes 4 /workspace/training/training/lora.py \
#   --model_name /workspace/training/models/gemma-3-12b-it-rewrite-v1029-epoch2-bf16 \
#   --train_path /workspace/training/datas/generation_data/train_bca-knowledge-qa-single-turn-epoch-3-b7_en_20251028_960.jsonl \
#   --save_model /workspace/training/models/gemma-3-12b-it-rewrite-v1029-epoch3-bf16 \
#   --learning_rate 0.00003

# export CUDA_VISIBLE_DEVICES=0,1,2,3 && accelerate launch --num_processes 4 /workspace/training/training/lora.py \
#   --model_name /workspace/training/models/gemma-3-12b-it-rewrite-v1029-epoch3-bf16 \
#   --train_path /workspace/training/datas/generation_data/train_bca-knowledge-qa-single-turn-epoch-4-b7_en_20251028_960.jsonl \
#   --save_model /workspace/training/models/gemma-3-12b-it-rewrite-v1029-epoch4-bf16 \
#   --learning_rate 0.00002

# export CUDA_VISIBLE_DEVICES=0,1,2,3 && accelerate launch --num_processes 4 /workspace/training/training/lora.py \
#   --model_name /workspace/training/models/gemma-3-12b-it-rewrite-v1029-epoch4-bf16 \
#   --train_path /workspace/training/datas/generation_data/train_bca-knowledge-qa-single-turn-epoch-5-b7_en_20251028_960.jsonl \
#   --save_model /workspace/training/models/gemma-3-12b-it-rewrite-v1029-epoch5-bf16 \
#   --learning_rate 0.00001
"""


import os
import json
import torch
import argparse
from datasets import load_dataset
from trl import SFTTrainer
from accelerate import PartialState
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training


os.environ['WANDB_MODE'] = 'dryrun'


ds_config_path = "./deepspeed_zero3.json"
ds_config = {
  "bf16": { "enabled": True },
  "fp16": { "enabled": False },
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": True,
    "contiguous_gradients": True,
    "reduce_bucket_size": 8388608,
    "stage3_prefetch_bucket_size": 4194304,
    "stage3_param_persistence_threshold": 1000000,
    "offload_optimizer": { "device": "none" },
    "offload_param": { "device": "none" }
  },
  "gradient_accumulation_steps": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
os.makedirs(os.path.dirname(ds_config_path), exist_ok=True)
with open(ds_config_path, "w", encoding="utf-8") as f:
    json.dump(ds_config, f, indent=2)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}")


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Script")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/training/models/results",
        help="训练输出目录"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/workspace/training/models/gemma-3-12b-it-sp-end-bf16",
        help="基座模型路径"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        nargs="+",  # 支持多个文件
        default=["/workspace/training/datas/generation_data/train_bca-knowledge-qa-single-turn-epoch-1-b7_en_20251028_960.jsonl"],
        help="训练数据路径（可传多个）"
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default="/workspace/training/models/gemma-3-12b-it-rewrite-v1029-epoch1-bf16",
        help="最终模型保存路径"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="最终模型保存路径"
    )
    return parser.parse_args()


# output_dir = "/workspace/training/models/results"
# model_name = "/workspace/training/models/gemma-3-12b-it-sp-end-bf16"
# train_path = [
#     "/workspace/training/datas/generation_data/train_bca-knowledge-qa-single-turn-epoch-1-b7_en_20251028_960.jsonl"
# ]
# save_model = "/workspace/training/models/gemma-3-12b-it-rewrite-v1029-epoch1-bf16"

args = parse_args()
output_dir = args.output_dir
model_name = args.model_name
train_path = args.train_path
save_model = args.save_model
learning_rate = args.learning_rate

print(args)
print(train_path)
print(save_model)
print(learning_rate)

adapter_name = "/workspace/training/models/results/final_checkpoint"

device_string = PartialState().process_index

dataset = load_dataset("json", data_files=train_path, split="train")
dataset = dataset.shuffle(seed=12345)

base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

tokenizer = AutoTokenizer.from_pretrained(model_name)


peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = get_peft_model(base_model, peft_config)
print_trainable_parameters(base_model)


def formatting_prompts_func(examples):
    output_texts = []
    for i in range(len(examples['dialog'])):
        message = examples['dialog'][i]
        new_message = []
        for j in message:
            new_message.append({"role": j["role"], "content": j["content"]})
        text = tokenizer.apply_chat_template(new_message, add_generation_prompt=True, tokenize=False)
        output_texts.append(text)
    return output_texts


def formatting_func(example):
    text = f"{example['input']}\n{example['instruction']}\n{example['output']}"
    return text


training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},  # 保存中间激活值
    max_grad_norm=0.3,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,
    save_total_limit=3,
    logging_steps=10,
    output_dir=output_dir,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05
)

trainer = SFTTrainer(
    base_model,
    train_dataset=dataset,
    max_seq_length=8192,
    formatting_func=formatting_prompts_func,
    args=training_args
)

trainer.train()
trainer.save_model(output_dir)

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


# 保存模型

model = AutoPeftModelForCausalLM.from_pretrained(adapter_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

new_model = model.merge_and_unload()
new_model.save_pretrained(save_model, torch_dtype=torch.bfloat16, safe_serialization=True)
tokenizer.save_pretrained(save_model)


preprocessor_config = {
  "crop_size": None,
  "data_format": "channels_first",
  "default_to_square": True,
  "device": None,
  "do_center_crop": None,
  "do_convert_rgb": None,
  "do_normalize": True,
  "do_pan_and_scan": None,
  "do_rescale": True,
  "do_resize": True,
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_processor_type": "Gemma3ImageProcessorFast",
  "image_seq_length": 256,
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "input_data_format": None,
  "pan_and_scan_max_num_crops": None,
  "pan_and_scan_min_crop_size": None,
  "pan_and_scan_min_ratio_to_activate": None,
  "processor_class": "Gemma3Processor",
  "resample": 2,
  "rescale_factor": 0.00392156862745098,
  "return_tensors": None,
  "size": {
    "height": 896,
    "width": 896
  }
}
with open(f"{save_model}/preprocessor_config.json", "w", encoding="utf-8") as f:
    json.dump(preprocessor_config, f, indent=2)
