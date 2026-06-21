import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. 基础配置与路径（与 artifacts/classification 最优折 epoch 表 + 复制的 .npy 对齐）
# ==========================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STEP3_OUT = os.path.join(_SCRIPT_DIR, "artifacts", "classification")
# 最优折 train / test 的 epoch 级清单（由 step3_classify_shap.py 生成）
TRAIN_EPOCH_CSV = os.path.join(STEP3_OUT, "best_fold_train_epochs.csv")
VAL_EPOCH_CSV = os.path.join(STEP3_OUT, "best_fold_test_epochs.csv")
# 与 step3 脚本复制特征目录一致：每个 subject_epoch 对应一个 .npy
TRAIN_FEAT_ROOT = os.path.join(_SCRIPT_DIR, "features", "fold_train")
VAL_FEAT_ROOT = os.path.join(_SCRIPT_DIR, "features", "fold_val")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "artifacts", "projector")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 本地 DeepSeek 权重目录（勿用普通字符串写 Windows 路径："\t" 会变成制表符导致报错）
MODEL_NAME_OR_PATH = os.path.join(os.path.dirname(_SCRIPT_DIR), "models", "deepseek-7b")

# 超参数 (加入梯度累加机制)
BATCH_SIZE = 2             # 物理 Batch Size (防爆显存，每次进显卡 2 个样本)
ACCUMULATION_STEPS = 4     # 梯度累加步数 (2 * 4 = 8，等效逻辑 Batch Size 为 8)
EPOCHS = 10
EARLY_STOP_PATIENCE = 3    # 验证 loss 连续若干 epoch 未低于历史最优则提前停止
LR = 1e-4                  # Projector 学习率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LLM_HIDDEN_SIZE = 3584     # Qwen2.5-7B/DeepSeek-R1-7B 的隐藏层维度
EEGPT_DIM = 512            # 你提取的脑电特征维度

# ==========================================
# 2. 定义投影层 (Projector) (无修改)
# ==========================================
class EEGProjector(nn.Module):
    def __init__(self, input_dim=EEGPT_DIM, hidden_dim=1024, output_dim=LLM_HIDDEN_SIZE):
        super(EEGProjector, self).__init__()
        # 使用双层 MLP 增强对非线性脑电特征的映射能力
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # 输入: (Batch, 512)
        # 输出: (Batch, 1, 3584) -> 将特征转为 LLM 认知中的 1 个外星 Token
        return self.proj(x).unsqueeze(1)

# ==========================================
# 3. 多模态数据集：epoch 级（artifacts/classification 导出的 best_fold_train/test_epoch.csv）
# ==========================================
class MultimodalEEGDataset(Dataset):
    def __init__(self, features_dir: str, epoch_csv_path: str):
        if not os.path.isfile(epoch_csv_path):
            raise FileNotFoundError(f"epoch CSV 不存在: {epoch_csv_path}")
        self.df = pd.read_csv(epoch_csv_path)
        if "epoch_report" not in self.df.columns:
            raise ValueError(f"CSV 缺少 epoch_report 列: {epoch_csv_path}")
        self.features_dir = features_dir

        self.samples = []
        for _, row in self.df.iterrows():
            subj = row["subject"]
            feat_path = os.path.join(self.features_dir, f"{subj}.npy")
            if not os.path.isfile(feat_path):
                continue
            feat = np.load(feat_path).reshape(-1, 512).mean(axis=0)
            report = row["epoch_report"]
            if pd.isna(report):
                report = ""

            text_prompt = (
                "你是一名顶级神经内科专家。请结合以下客观数据与隐式特征向量，判断该受试者是否患有抑郁症(MDD)。\n"
                f"{report}\n"
                "【连续脑电隐式向量输入】：\n"
            )

            lab = int(row["label"]) if not pd.isna(row["label"]) else 0
            label_text = "MDD" if lab == 1 else "HC"

            self.samples.append(
                {
                    "eeg_feat": torch.tensor(feat).float(),
                    "prompt_text": text_prompt,
                    "label_text": label_text,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ==========================================
# 4. 验证函数 (新增：监控验证集损失)
# ==========================================
def validate_projector(projector, llm, tokenizer, val_dataloader):
    projector.eval()  # 投影层切评估模式
    total_val_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating"):
            eeg_feats = batch["eeg_feat"].to(DEVICE)
            prompt_texts = batch["prompt_text"]
            label_texts = batch["label_text"]
            
            # 文本转Tokens
            prompt_tokens = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(DEVICE)
            label_tokens = tokenizer(label_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
            
            # 提取文本Embeddings
            prompt_embeds = llm.get_input_embeddings()(prompt_tokens.input_ids)
            label_embeds = llm.get_input_embeddings()(label_tokens.input_ids)
            
            # 脑电特征投影
            eeg_proj_embeds = projector(eeg_feats).to(torch.bfloat16)
            
            # 拼接输入
            inputs_embeds = torch.cat([prompt_embeds, eeg_proj_embeds, label_embeds], dim=1)
            
            # 构建遮罩Labels
            batch_size_current = inputs_embeds.size(0)
            seq_len_prompt = prompt_tokens.input_ids.size(1)
            seq_len_eeg = 1
            targets = torch.full((batch_size_current, inputs_embeds.size(1)), -100, dtype=torch.long).to(DEVICE)
            start_idx = seq_len_prompt + seq_len_eeg
            targets[:, start_idx:] = label_tokens.input_ids
            
            # 前向传播计算损失
            outputs = llm(inputs_embeds=inputs_embeds, labels=targets)
            total_val_loss += outputs.loss.item()
    
    avg_val_loss = total_val_loss / len(val_dataloader)
    projector.train()  # 切回训练模式
    return avg_val_loss

# ==========================================
# 5. 模型加载与训练主逻辑 (核心修改：加入验证、最优模型保存)
# ==========================================
def train_projector():
    cfg_json = os.path.join(MODEL_NAME_OR_PATH, "config.json")
    if not os.path.isfile(cfg_json):
        raise FileNotFoundError(
            f"本地模型目录无效（缺少 config.json）: {MODEL_NAME_OR_PATH}\n"
            "请确认权重已解压到与 step_4.py 同级的 deepseek_model 文件夹。"
        )

    print("🔄 正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("🚀 正在以 4-bit 量化加载 LLM (极其节省显存)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    
    # 【核心防御】：彻底冻结大模型，防止医学常识被小样本破坏
    for param in llm.parameters():
        param.requires_grad = False
    llm.eval()

    print("🧠 初始化 Projector 投影层...")
    projector = EEGProjector().to(DEVICE)
    projector.train() 
    
    optimizer = optim.AdamW(projector.parameters(), lr=LR)
    
    # 构建训练/验证数据集和加载器 (核心修改：加载step3的train/val集)
    print("📥 加载训练集...")
    print(f"    CSV: {TRAIN_EPOCH_CSV}\n    特征目录: {TRAIN_FEAT_ROOT}")
    train_dataset = MultimodalEEGDataset(TRAIN_FEAT_ROOT, TRAIN_EPOCH_CSV)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    print("📥 加载验证集...")
    print(f"    CSV: {VAL_EPOCH_CSV}\n    特征目录: {VAL_FEAT_ROOT}")
    val_dataset = MultimodalEEGDataset(VAL_FEAT_ROOT, VAL_EPOCH_CSV)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    if len(train_dataset) == 0:
        raise RuntimeError(
            "训练样本数为 0：请检查 TRAIN_FEAT_ROOT 下是否有与 CSV 中 subject 同名的 .npy（由 artifacts/classification 复制到 artifacts/classificationtrain）。"
        )
    if len(val_dataset) == 0:
        raise RuntimeError(
            "验证样本数为 0：请检查 VAL_FEAT_ROOT（artifacts/classificationval）与 VAL_EPOCH_CSV 是否一致。"
        )

    print("\n" + "="*50)
    print(f"🔥 开始跨模态对齐训练 (适配step3最优折)")
    print(f"📦 训练集大小: {len(train_dataset)} | 验证集大小: {len(val_dataset)}")
    print(f"⚙️  物理 Batch Size: {BATCH_SIZE} | 累加步数: {ACCUMULATION_STEPS} | 逻辑 Batch: {BATCH_SIZE * ACCUMULATION_STEPS}")
    print("="*50)

    # 新增：跟踪最优验证损失，保存最优模型
    best_val_loss = float('inf')
    best_model_path = os.path.join(OUTPUT_DIR, "best_eeg_projector.pth")
    epochs_no_improve = 0  # 连续未刷新最优验证 loss 的 epoch 数

    for epoch in range(EPOCHS):
        total_train_loss = 0
        optimizer.zero_grad() # 每个 Epoch 开始前清空梯度
        
        # 训练阶段
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")):
            eeg_feats = batch["eeg_feat"].to(DEVICE)
            prompt_texts = batch["prompt_text"]
            label_texts = batch["label_text"]
            
            # 1. 文本转 Tokens
            prompt_tokens = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(DEVICE)
            label_tokens = tokenizer(label_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
            
            # 2. 提取文本原生的 Embeddings (无梯度)
            with torch.no_grad():
                prompt_embeds = llm.get_input_embeddings()(prompt_tokens.input_ids)
                label_embeds = llm.get_input_embeddings()(label_tokens.input_ids)
            
            # 3. 翻译脑电特征 (有梯度)
            eeg_proj_embeds = projector(eeg_feats).to(torch.bfloat16) 
            
            # 4. 完美拼接: [文本Prompt] + [脑电向量] + [标签]
            inputs_embeds = torch.cat([prompt_embeds, eeg_proj_embeds, label_embeds], dim=1)
            
            # 5. 构建遮罩 Labels: 让模型只在乎最后的分类结果
            batch_size_current = inputs_embeds.size(0)
            seq_len_prompt = prompt_tokens.input_ids.size(1)
            seq_len_eeg = 1
            
            targets = torch.full((batch_size_current, inputs_embeds.size(1)), -100, dtype=torch.long).to(DEVICE)
            start_idx = seq_len_prompt + seq_len_eeg
            targets[:, start_idx:] = label_tokens.input_ids
            
            # 6. 前向传播
            outputs = llm(inputs_embeds=inputs_embeds, labels=targets)
            
            # 7. 梯度累加计算
            loss = outputs.loss / ACCUMULATION_STEPS 
            loss.backward()
            
            # 如果攒够了步数，或者是这个 Epoch 的最后一个 batch，执行更新
            if ((i + 1) % ACCUMULATION_STEPS == 0) or (i + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()
            
            # 记录真实 Loss 用于展示
            total_train_loss += outputs.loss.item() 
        
        # 计算训练集平均损失
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # 验证阶段
        avg_val_loss = validate_projector(projector, llm, tokenizer, val_dataloader)
        
        # 保存最优模型 + 早停计数（「不降」= 未严格低于历史最优验证 loss）
        if avg_val_loss < best_val_loss:
            prev_best = best_val_loss
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(projector.state_dict(), best_model_path)
            print(f"📌 最优模型更新！验证损失: {prev_best:.4f} → {avg_val_loss:.4f}，已保存至 {best_model_path}")
        else:
            epochs_no_improve += 1

        print(f"✅ Epoch {epoch+1} 完成 | 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f}")
        print(f"📈 当前最优验证损失: {best_val_loss:.4f} | 验证无改善: {epochs_no_improve}/{EARLY_STOP_PATIENCE}")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(
                f"⏹️ 验证损失已连续 {EARLY_STOP_PATIENCE} 个 epoch 未下降（相对历史最优），提前结束训练。"
            )
            break

    # 保存最后一轮模型（兜底）
    final_model_path = os.path.join(OUTPUT_DIR, "eeg_projector_final.pth")
    torch.save(projector.state_dict(), final_model_path)
    print(f"\n🎉 跨模态对齐训练完成！")
    print(f"📊 最优模型路径: {best_model_path} (验证损失: {best_val_loss:.4f})")
    print(f"📋 最后一轮模型路径: {final_model_path}")

if __name__ == "__main__":
    train_projector()