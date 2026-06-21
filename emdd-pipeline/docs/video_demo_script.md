# E-MDD 招聘演示视频 — 操作流程稿

> **建议总时长：** 8～12 分钟（精简版 5～6 分钟见文末）  
> **录制形式：** 屏幕录制 + 可选摄像头小窗  
> **原则：** 展示「架构 + 编排能力 + 可解释输出 + 脱敏指标」，不暴露原始 EEG、完整 CoT、本机绝对路径

---

## 录制前准备清单

- [ ] 仓库已在 GitHub 上（或本地干净 clone），README 无 `D:/` 等本机路径
- [ ] 终端字体放大（16～18pt），主题对比度清晰
- [ ] 准备好 **脱敏** 材料（可本地打开，不要录进真实路径）：
  - `docs/architecture.md` 或 README 架构图
  - `docs/sample_outputs.md` 指标表
  - 1 段匿名 CoT 片段（来自 sample_outputs 或自行打码）
  - 可选：step3 epoch 报告截图（去掉被试 ID）
- [ ] 本机已能运行：`python emdd_workflow.py list`（至少 step3～5 显示 DONE 更佳）
- [ ] 关闭通知、隐藏桌面敏感文件夹名

---

## 第一幕：开场与项目定位（约 45 秒）

### 展示什么

1. GitHub 仓库首页 → 滚动到 README 标题 **E-MDD Pipeline**
2. 短暂停留在「亮点（招聘向摘要）」bullet 列表

### 说什么（口播稿）

> 大家好，这个项目是我做的 **E-MDD**——Explainable MDD diagnosis，一个可解释的抑郁症脑电诊断 pipeline。  
> 它不是单纯的 EEGPT 微调，而是端到端架构：从预处理、EEGPT 编码、分类与 SHAP 可解释性，到 LLM 的 Chain-of-Thought 推理。  
> 仓库公开的是 **workflow 编排和代码结构**；数据和权重因体积与许可没有上传，但我会演示如何验证整条链路。

### 要点

- 强调架构名 **E-MDD**，EEGPT 只是 step1 编码器
- 一句话说明「公开代码、不公开数据」是合理设计

---

## 第二幕：架构与数据流（约 1.5 分钟）

### 展示什么

1. README 中的 **Mermaid 流程图**（或 `docs/architecture.md` 数据流）
2. Step 表格：`step0` → `step1` → … → `eval`
3. 可选：快速打开 `workflow/` 目录树（`config.py`、`runner.py`、`adapters/`）

### 说什么

> 整条链路分七步。  
> **step0** 是 MATLAB 预处理：6 秒滑窗、按被试划分 train/test，可选 Z-norm。  
> **step1** 用预训练 EEGPT 把每个 epoch 编成 512 维向量。  
> **step2** 用 MNE 算 PSD 和 alpha 不对称等物理指标。  
> **step3** 是 E-MDD 核心分类头，配合 **StratifiedGroupKFold** 做被试级五折，避免泄露；同时产出 epoch 级和 subject 级 SHAP 报告。  
> **step4** 训练一个两层 MLP，把 EEG 特征投影到 LLM 的 hidden space。  
> **step5** 冻结 7B 模型，生成结构化 CoT。  
> 最后 **eval** 在外部测试集上算 AUC 和 subject 准确率。  
> 我把这些步骤统一收进 `workflow` 包，用 YAML 配置驱动，而不是散落的硬编码路径。

### 要点

- 提到 **防泄露**（被试级 CV）和 **三层解释**（SHAP / 物理特征 / CoT）
- 若时间紧，跳过打开 `workflow/` 目录

---

## 第三幕：配置与可移植性（约 1 分钟）

### 展示什么

1. `configs/emdd_default.yaml` — 滚动 `paths:` 段，指出都是相对路径如 `emdd_core/features`、`data/eeg/train_6`
2. `configs/emdd_local.yaml.example` — 说明复制为本机配置
3. `data/README.md`、`vendor/README.md` 各扫一眼

### 说什么

> 配置上我刻意做了分层：  
> **default** 里是相对仓库根的路径，任何人 clone 都能看懂结构；  
> **local** 文件 gitignore，放本机的 EEG 目录、EEGPT checkpoint、LLM 权重路径。  
> 数据和模型在 `data/`、`vendor/`、`models/` 下，README 里写了怎么准备。  
> 这样 GitHub 上不会出现我电脑的 D 盘路径，但 collaborator 或面试官能按文档复现环境。

### 要点

- 体现工程化思维：**可移植 + 私密路径隔离**
- **不要**打开真实的 `emdd_local.yaml`（若含绝对路径）

---

## 第四幕：Workflow CLI 演示（约 2.5 分钟）★核心

### 展示什么

在仓库根目录终端依次执行：

```bash
python emdd_workflow.py list
python emdd_workflow.py check --all
python emdd_workflow.py run --from step3 --dry-run
```

可选第四条（若 step3 已完成且不太耗时）：

```bash
python emdd_workflow.py run --steps step3 --skip-existing
```

然后打开最近一次 `runs/<timestamp>/`：

- `command.txt`
- `config.resolved.yaml`
- `run_meta.json`
- `logs/step3.log`（前几行即可）

### 说什么

> 入口是 `emdd_workflow.py`。  
> **`list`** 会列出每一步的 DONE 或 PENDING，以及缺什么输入——这是我设计的 artifact 契约。  
> **`check`** 做运行前检查，避免跑到一半才发现缺特征或缺权重。  
> **`dry-run`** 只打印将要执行的命令，适合演示编排逻辑而不真的训几小时。  
> 每次真实运行都会在 `runs/` 下留档：命令、解析后的配置、环境信息和分步日志，方便复现和排错。  
> 这证明我不仅能写模型脚本，还能把 **多步实验 pipeline 产品化**。

### 要点

- `list` 输出里指着 **step3 / step4 / step5** 的 DONE 说「下游我已跑通」
- step1/2 PENDING 可以说「缺公开特征文件，预期行为」
- dry-run 是招聘演示的**安全命令**，不会误触发长训练

---

## 第五幕：模型与可解释性（约 2 分钟）

### 展示什么

1. `docs/architecture.md` 中 **MDDClassificationHead** 和 **EEGProjector** 结构（512→256→2 与 512→1024→3584）
2. 脱敏的 step3 产物说明（二选一）：
   - 打开 `emdd_core/artifacts/classification/` 下某个 **epoch 报告** CSV 的列名（不要滚动到含真实 ID 的行），或
   - 展示事先准备的打码截图
3. `docs/sample_outputs.md` 中的 **CoT 片段示例**

### 说什么

> 分类头是一个两层 MLP，输入 512 维 EEGPT 特征，输出 MDD 概率。  
> 训练时用加权交叉熵，验证用被试级分组。  
> SHAP 这边我同时做了 epoch 级和 subject 级报告，并把重要维度映射回 alpha 功率、不对称性等 **可解释的物理量**。  
> step4 的投影层把 EEG 压成 LLM 能消费的一个 token embedding，step5 再让模型生成三段式 CoT：整体判断、TOP3 特征推理、一致性结论。  
> 这里是脱敏后的 CoT 样例——可以看到它会把 SHAP TOP 特征和临床术语对齐，而不是黑盒给一个分数。

### 要点

- 强调 **XAI + LLM** 是差异化能力
- 画面里 **不要**出现完整被试编号或原始 EEG 波形（除非已匿名）

---

## 第六幕：实验结果（约 1 分钟）

### 展示什么

1. README 或 `docs/sample_outputs.md` 的结果表格
2. 可选：外部测试混淆矩阵一行（TN/FP/FN/TP）

### 说什么

> 在外部测试集上，17 个被试、1702 个 epoch，EEGPT 加临床特征 AUC 约 **0.85**，subject 准确率约 **0.82**。  
> 我也跑了 EEGNet 和 EEG-Conformer 作为对照——EEGNet epoch 级 AUC 更高，但 subject 级我们方法更稳。  
> 完整 CSV 和权重没有公开，但指标和 workflow 日志可以交叉验证我确实跑通过。

### 要点

- 数字与 `sample_outputs.md` **保持一致**
- 诚实提 baseline 对比，显得严谨

---

## 第七幕：MATLAB 预处理（可选，约 45 秒）

> 若总时长要控制在 8 分钟内，**整幕可删**。

### 展示什么

`preprocessing/matlab/README.md` + `emdd_run_preprocess.m` 文件名即可

### 说什么

> step0 我把 MATLAB 脚本也接进了 workflow，包括 6 秒 epoch、按被试划分和可选 Z-norm。  
> ICA 脚本因为依赖本机 EEGLAB 插件，通过 local 配置挂载，默认关闭。  
> 如果已有预处理好的 train_6，可以直接从 step1 开始——这也是真实实验里的常见情况。

---

## 第八幕：收尾（约 30 秒）

### 展示什么

回到 GitHub README 底部「公开范围说明」+ 你的联系方式（简历/GitHub/邮箱）

### 说什么

> 总结一下：这个仓库展示的是 **E-MDD 全链路架构设计、workflow 编排、可解释 AI 和 LLM 融合**。  
> 数据和模型按合规与体积做了隔离，但 list、check、dry-run 和 runs 日志可以验证工程完整度。  
> 欢迎看代码里的 `workflow/adapters` 和文档，也欢迎交流。谢谢。

---

## 精简版时间轴（5～6 分钟）

| 时间 | 内容 | 命令/文件 |
|------|------|-----------|
| 0:00 | 开场 + 亮点 | README |
| 0:45 | 架构图 60 秒 | README mermaid |
| 1:45 | 配置分层 45 秒 | emdd_default.yaml + example |
| 2:30 | **list + check + dry-run** | 终端 |
| 4:00 | CoT 样例 + 指标 | sample_outputs.md |
| 5:00 | 收尾 | README |

---

## 常见问题（录制时可能被问）

| 问题 | 建议回答 |
|------|----------|
| 为什么没有数据？ | Mumtaz 公开数据集 + 大体积权重；仓库提供 workflow 与复现说明 |
| 标签怎么来的？ | 数据集文件名规则 MDD/HC，无本项目内人工标注 |
| 和 EEGPT 论文关系？ | EEGPT 是 step1 编码器；E-MDD 是整体架构与下游 |
| 能否一键复现？ | 准备 data/vendor/models 后 `run --all`；GPU 与 LLM 步骤需本机资源 |
| 如何证明是你做的？ | runs 日志、config 设计、adapter 代码、脱敏指标与 CoT 结构一致性 |

---

## 录制技巧

1. **先录终端三段命令**，再补录口播，后期剪辑对齐
2. 鼠标移动慢、停 1～2 秒再点击
3. 口播用「我设计了…」「这一步解决…泄露/可解释…」比「这个文件是…」更有招聘感
4. 导出 1080p；码率不必极高，字幕建议开（中文）
5. 视频描述里贴：GitHub 链接 + 三行 bullet（E-MDD / workflow / XAI+CoT）

---

## 视频描述模板（复制到 B 站/YouTube/领英）

```
E-MDD: Explainable MDD Diagnosis Pipeline
- 端到端脑电 pipeline：MATLAB 预处理 → EEGPT → SHAP → LLM CoT
- 自研 workflow CLI：list / check / dry-run / 运行日志归档
- 被试级 5-fold CV + 脱敏外部测试 AUC 0.85

GitHub: <your-repo-url>
代码公开；数据与权重见仓库 README。
```
