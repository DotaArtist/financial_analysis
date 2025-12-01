import os
import torch
import torch.distributed.checkpoint as dist_cp
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 配置区域 =================
BASE_MODEL_PATH = "/workspace/training/pretrained_model/gemma-3-12b-it-sp-end"
CHECKPOINT_DIR = "/workspace/training/models/dpo_gemma3_12b_it_sp_end/checkpoint-3318"
SAVE_PATH = "/workspace/training/models/gemma-3-12b-it-sp-end-dpov1130"
# ===========================================

def inspect_checkpoint_keys(fsdp_path):
    """读取 Checkpoint 元数据，返回所有存在的 key"""
    print(f"🔍 正在读取 Checkpoint 元数据: {fsdp_path}")
    try:
        reader = dist_cp.FileSystemReader(fsdp_path)
        metadata = reader.read_metadata()
        return set(metadata.state_dict_metadata.keys())
    except Exception as e:
        print(f"❌ 读取元数据失败: {e}")
        return set()

def main():
    print(f"1. 初始化基础模型: {BASE_MODEL_PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # 获取模型期望的 Key
    model_keys = set(model.state_dict().keys())

    fsdp_weights_path = os.path.join(CHECKPOINT_DIR, "pytorch_model_fsdp_0")
    if not os.path.exists(fsdp_weights_path):
        raise FileNotFoundError(f"找不到权重路径: {fsdp_weights_path}")

    # 2. 获取 Checkpoint 中实际存在的 Key
    checkpoint_keys = inspect_checkpoint_keys(fsdp_weights_path)

    if not checkpoint_keys:
        print("❌ 无法获取 Checkpoint keys，脚本终止。")
        return

    print(f"📊 统计: 模型期望 {len(model_keys)} 个参数, Checkpoint 包含 {len(checkpoint_keys)} 个参数")

    # 3. 构建加载字典 (Mapping)
    state_dict_to_load = {}

    # =======================================================
    # 核心修复：更新匹配逻辑，处理 'model.' 前缀
    # =======================================================
    def find_matching_key(target_key, ckpt_keys):
        # 1. 尝试直接匹配
        if target_key in ckpt_keys:
            return target_key

        # 2. 【修复点】尝试添加 'model.' 前缀
        # 你的日志显示 checkpoint 里的 key 多了 'model.' 开头
        prefix_key = "model." + target_key
        if prefix_key in ckpt_keys:
            return prefix_key

        # 3. 尝试其他常见的 FSDP 前缀
        if ("_fsdp_wrapped_module." + target_key) in ckpt_keys:
            return "_fsdp_wrapped_module." + target_key

        return None
    # =======================================================

    print("🛠️  正在构建参数映射...")

    mapped_count = 0
    missing_keys = []

    original_state_dict = model.state_dict()

    for model_key, tensor in original_state_dict.items():
        # 跳过 lm_head (Weight Tying 问题)，防止 FSDP 加载报错
        if "lm_head.weight" in model_key:
            continue

        found_key = find_matching_key(model_key, checkpoint_keys)

        if found_key:
            # 建立映射: Checkpoint Key -> Model Tensor
            state_dict_to_load[found_key] = tensor
            mapped_count += 1
        else:
            missing_keys.append(model_key)

    print(f"✅ 成功映射 {mapped_count} 个参数。")

    if missing_keys:
        print(f"⚠️  以下 {len(missing_keys)} 个参数在 Checkpoint 中未找到 (前5个):")
        for k in missing_keys[:5]:
            print(f"   - {k}")

    if mapped_count == 0:
        print("❌ 依然没有匹配成功，请检查脚本逻辑。")
        return

    # 4. 执行加载
    print("🚀 开始加载权重 (dist_cp.load)...")
    # 注意：dist_cp.load 会直接把数据写入 state_dict_to_load 的 values (即 model 的 tensor)
    dist_cp.load(
        state_dict=state_dict_to_load,
        checkpoint_id=fsdp_weights_path,
    )

    # 5. 手动修复 Weight Tying (lm_head)
    print("🔗 重新绑定 lm_head 权重...")
    try:
        # Gemma 3 结构通常是 model.language_model.model.embed_tokens
        # 但我们这里操作的是 AutoModel 加载的对象
        if hasattr(model, "language_model"):
            model.language_model.lm_head.weight = model.language_model.model.embed_tokens.weight
        elif hasattr(model, "lm_head"):
            model.lm_head.weight = model.model.embed_tokens.weight
        print("   -> 绑定成功")
    except Exception as e:
        print(f"   -> [警告] 自动绑定失败，请确认模型结构: {e}")

    # 6. 保存
    print(f"💾 保存 Safetensors 至: {SAVE_PATH}")
    model.save_pretrained(SAVE_PATH, safe_serialization=True, max_shard_size="5GB")

    try:
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR, trust_remote_code=True)
    except:
        print("Checkpoint 中无 tokenizer，从 Base Model 复制...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    tokenizer.save_pretrained(SAVE_PATH)

    print("✨ 全部完成！")

if __name__ == "__main__":
    main()
