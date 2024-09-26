import os
import torch
from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

os.environ['WANDB_MODE'] = 'dryrun'

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return list(lora_module_names)


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
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
    )


output_dir = "./results"
model_name = "/mnt/ant-cc/yewen.yp/Qwen2-0.5B-Instruct/"

train_path = [
    "/ossfs/node_42404819/workspace/data/train_json/train_v911.jsonl"
]

device_string = PartialState().process_index

dataset = load_dataset("json", data_files=train_path, split="train")
dataset = dataset.shuffle(seed=12345)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                  device_map='auto')
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
target_modules = find_all_linear_names(base_model)
print(target_modules)

# Change the LORA hyperparameters accordingly to fit your use case
peft_config = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = get_peft_model(base_model, peft_config)
print_trainable_parameters(base_model)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['tag_attribute'])):
        tmp = {
            "业务属性": example['tag_attribute'][i],
            "业务场景": example['tag_business'][i],
            "历史咨询": example['pre_ask'][i],
            "用户问题": example['ask'][i] if "？" in example['ask'][i] else example['ask'][i] + "？"
            }

        text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{str(tmp)}<|im_end|>\n<|im_start|>assistant\n{example['question_name'][i]}<|im_end|>\n"
        output_texts.append(text)
    return output_texts

def formatting_func(example):
    text = f"{example['input']}\n{example['instruction']}\n{example['output']}"
    return text

training_args = TrainingArguments(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},  # 保存中间激活值
    max_grad_norm=0.3,
    num_train_epochs=3,
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
    tokenizer=tokenizer,
    max_seq_length=2048,
    formatting_func=formatting_prompts_func,
    args=training_args
)

trainer.train()
trainer.save_model(output_dir)

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM

adapter_name = ""
model_name = ""

model = AutoPeftModelForCausalLM.from_pretrained(adapter_name, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)

new_model = model.merge_and_unload()
new_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
