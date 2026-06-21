# 第三方依赖

## EEGPT 编码器（step1）

```bash
git clone https://github.com/wzhlearning/EEGPT.git vendor/EEGPT
```

下载 checkpoint `eegpt_mcae_58chs_4s_large4E.ckpt` 至：

```
vendor/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt
```

Figshare: https://figshare.com/s/e37df4f8a907a866df4b

- 默认配置 [`configs/emdd_default.yaml`](../configs/emdd_default.yaml) 使用相对仓库根的路径；本机覆盖见 [`configs/emdd_local.yaml.example`](../configs/emdd_local.yaml.example)

## EEG-Conformer（对照实验，可选）

若运行 `0422/eegconformer_512_pipeline.py`，请自行克隆 EEG-Conformer 并在脚本或本地配置中指定路径。
