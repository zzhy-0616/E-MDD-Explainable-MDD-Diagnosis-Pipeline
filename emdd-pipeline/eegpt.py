"""
从 vendor/EEGPT 加载 EEGPT 模型（路径可在 configs/emdd_local.yaml 覆盖）。

预训练权重需手动下载：
https://figshare.com/s/e37df4f8a907a866df4b
文件：eegpt_mcae_58chs_4s_large4E.ckpt
默认放置：vendor/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt
"""
import sys
import os
from functools import partial
import torch

# EEGPT 项目路径（相对仓库根；推荐通过 emdd_workflow step1 从配置注入）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EEGPT_ROOT = os.path.join(_SCRIPT_DIR, "vendor", "EEGPT")
EEGPT_DOWNSTREAM_TUEG = os.path.join(EEGPT_ROOT, "downstream_tueg")

# 添加路径以便导入
for p in [EEGPT_DOWNSTREAM_TUEG, EEGPT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 默认预训练 checkpoint 路径
DEFAULT_CKPT_PATH = os.path.join(EEGPT_ROOT, "checkpoint", "eegpt_mcae_58chs_4s_large4E.ckpt")


def print_checkpoint_info(checkpoint_path=None):
    """
    打印 checkpoint 权重信息：文件大小、参数量、各层张量形状等。
    """
    ckpt = checkpoint_path or DEFAULT_CKPT_PATH
    if not os.path.isfile(ckpt):
        print(f"文件不存在: {ckpt}")
        return

    # 文件大小
    file_size_bytes = os.path.getsize(ckpt)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"【文件】 {ckpt}")
    print(f"  磁盘占用: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")

    # 加载 state_dict
    ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = ckpt_data.get("state_dict", ckpt_data)
    if isinstance(state, dict):
        state = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    else:
        state = {}

    # 参数量统计
    total_params = 0
    print(f"\n【权重】 共 {len(state)} 个张量")
    for name, t in sorted(state.items()):
        n = t.numel()
        total_params += n
        size_mb = n * t.element_size() / (1024 * 1024)
        print(f"  {name}: shape {tuple(t.shape)} | {n:,} params | {size_mb:.4f} MB")

    total_mb = total_params * 4 / (1024 * 1024)  # float32
    print(f"\n【总计】 参数量: {total_params:,} | 约 {total_mb:.2f} MB (float32)")


def load_eegpt_encoder(checkpoint_path=None, device=None):
    """
    仅加载 EEGPT 编码器（EEGTransformer），用于特征提取。

    预训练配置：58 通道，256Hz，4 秒 = 1024 样本，patch_size 64。

    Args:
        checkpoint_path: .ckpt 文件路径，不传则返回随机初始化模型。
        device: 设备，默认 cuda 若可用否则 cpu。

    Returns:
        model: EEGTransformer 编码器，输入形状 (B, C, T)，如 (B, 58, 1024)。
    """
    from Modules.models.EEGPT_mcae_finetune_change import EEGTransformer

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 预训练时的配置：58ch, 1024 samples (4s @ 256Hz)
    model = EEGTransformer(
        img_size=[19, 1024],
        patch_size=64,
        embed_dim=512,
        embed_num=4,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_std=0.02,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
    )
    model = model.to(device)
    model.eval()

    ckpt = checkpoint_path or DEFAULT_CKPT_PATH
    if os.path.isfile(ckpt):
        ckpt_data = torch.load(ckpt, map_location=device, weights_only=False)
        state = ckpt_data.get("state_dict", ckpt_data)
        # 提取 target_encoder 部分
        encoder_state = {k[15:]: v for k, v in state.items() if k.startswith("target_encoder.")}
        if encoder_state:
            model.load_state_dict(encoder_state, strict=False)
            print(f"已从 {ckpt} 加载编码器权重")
    elif checkpoint_path:
        print(f"未找到 {checkpoint_path}，使用随机初始化")

    return model


def load_eegpt_classifier(
    num_classes=4,
    in_channels=22,
    img_size=None,
    use_channels_names=None,
    checkpoint_path=None,
    device=None,
):
    """
    加载完整 EEGPTClassifier（编码器 + 分类头），用于下游分类。

    Args:
        num_classes: 分类数。
        in_channels: 输入通道数。
        img_size: [C, T]，不传则用 [19, 2000]（与分类头 63488 维匹配）。
        use_channels_names: 通道名列表，需在 CHANNEL_DICT 中。不传则用 19 导 10-20。
        checkpoint_path: .ckpt 路径，不传则尝试默认路径。
        device: 设备。

    Returns:
        model: EEGPTClassifier。
    """
    from Modules.models.EEGPT_mcae_finetune_change import EEGPTClassifier
    from utils import load_state_dict

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if img_size is None:
        img_size = [19, 2000]  # 与分类头 63488 匹配（31 time patches）
    if use_channels_names is None:
        use_channels_names = [
            "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
            "T7", "C3", "CZ", "C4", "T8",
            "P7", "P3", "PZ", "P4", "P8",
            "O1", "O2",
        ]

    model = EEGPTClassifier(
        num_classes=num_classes,
        in_channels=in_channels,
        img_size=img_size,
        use_channels_names=use_channels_names,
        use_chan_conv=True,
    )
    model = model.to(device)
    model.eval()

    ckpt = checkpoint_path or DEFAULT_CKPT_PATH
    if os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
        ckpt_model = checkpoint.get("state_dict", checkpoint)
        load_state_dict(model, ckpt_model, prefix="")
        print(f"已从 {ckpt} 加载分类器权重")
    elif checkpoint_path:
        print(f"未找到 {checkpoint_path}，使用随机初始化")

    return model


if __name__ == "__main__":
    # 打印 checkpoint 权重信息
    print_checkpoint_info()
    print("\n" + "=" * 60 + "\n")

    # 示例 1：仅编码器（特征提取），输入 (B, 58, 1024)
    encoder = load_eegpt_encoder()
    x = torch.randn(2, 19, 1024).to(next(encoder.parameters()).device)
    with torch.no_grad():
        z = encoder(x)  # chan_ids 默认 None，内部会用 arange
    print("Encoder 输出形状:", z.shape)

    # 示例 2：完整分类器（需 in_channels 通道，时间长度与 img_size[1] 一致）
    # model = load_eegpt_classifier(num_classes=4, in_channels=22, img_size=[19, 2000])
    # x = torch.randn(2, 22, 2000).to(next(model.parameters()).device)
    # with torch.no_grad():
    #     logits = model(x)
    # print("Logits 形状:", logits.shape)
