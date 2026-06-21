# 模型权重目录（不提交 Git）

## DeepSeek / LLM（step4、step5）

将本地 7B 指令模型解压到此目录，默认配置路径：

```
models/deepseek-7b/
├── config.json
├── tokenizer.json
└── ...
```

在 `configs/emdd_local.yaml` 中可通过 `paths.deepseek_model` 覆盖。

## 说明

- step4/5 训练与推理需要 GPU 与较大显存；公开仓库仅包含脚本与 workflow 编排。
- 训练产出的投影层权重位于 `emdd_core/artifacts/projector/`（已 gitignore）。
