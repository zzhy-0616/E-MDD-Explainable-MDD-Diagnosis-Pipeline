"""
从 vendor/ 或本地路径加载 EEG-Conformer 模型（对照实验，可选）。

请自行克隆 EEG-Conformer 并在本地配置或脚本中指定 root。
"""
import sys
import os
import torch
import torch.nn as nn

# 添加 EEG-Conformer 项目路径，以便导入模型
EEG_CONFORMER_ROOT = r"D:\1\EEG-Conformer-main\EEG-Conformer-main"
if EEG_CONFORMER_ROOT not in sys.path:
    sys.path.insert(0, EEG_CONFORMER_ROOT)

# 从原项目导入 Conformer 模型类
from conformer import Conformer


def load_conformer(
    checkpoint_path=None,
    emb_size=40,
    depth=6,
    n_classes=4,
    device=None,
    strict=True,
):
    """
    加载 EEG-Conformer 模型。

    Args:
        checkpoint_path: 可选，.pth 权重文件路径（例如该目录下的 model.pth）。
                        不传则返回未加载权重的模型。
        emb_size: 嵌入维度，默认 40，需与训练时一致。
        depth: Transformer 层数，默认 6。
        n_classes: 分类数，默认 4（BCI IV 2a）。
        device: 设备，默认 cuda 若可用否则 cpu。
        strict: 加载 state_dict 时是否严格匹配键，默认 True。

    Returns:
        model: 加载好的 Conformer 模型（已 eval）。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Conformer(emb_size=emb_size, depth=depth, n_classes=n_classes)
    model = model.to(device)

    if checkpoint_path and os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        # 若保存时用了 DataParallel，键可能带 "module." 前缀
        if list(state.keys())[0].startswith("module."):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=strict)
        model.eval()
        print(f"已从 {checkpoint_path} 加载权重")
    else:
        model.eval()
        if checkpoint_path:
            print(f"未找到权重文件 {checkpoint_path}，使用随机初始化模型")

    return model


# 使用示例
if __name__ == "__main__":
    # 仅加载模型结构（无预训练权重）
    model = load_conformer()
    print("模型结构已加载（随机初始化）")

    # 若有训练好的 .pth 文件，可指定路径加载权重，例如：
    # weight_file = r"D:\1\EEG-Conformer-main\EEG-Conformer-main\model.pth"
    # model = load_conformer(checkpoint_path=weight_file)

    # 推理示例（输入形状: batch, 1, 22, 1000）
    # x = torch.randn(2, 1, 22, 1000).to(next(model.parameters()).device)
    # with torch.no_grad():
    #     tok, logits = model(x)
    # print("logits shape:", logits.shape)
