# 五子棋神经网络训练指南 (Gomoku Neural Network Training Guide)

本指南说明了如何启动和配置五子棋深度学习模型的训练。训练 pipeline 融合了 Rapfi 论文的网络架构（CoordConv、边界距离平面、可学习位置编码、多任务辅助头）与 D4 对称数据增强、边缘重采样等技术。

---

## 1. 代码完善度与近期修正说明

在启动正式训练前，我们对代码库进行了深度的静态分析、逻辑审查与运行优化：
- **逻辑缺陷修正**：
  - **验证集温度一致性**：修复了验证集评估时漏掉 softmax 温度缩放的 Bug，保证了训练与评估时的 Cross-Entropy Loss 在数学上完全一致。修复后验证集 Loss 表现与训练阶段完美契合。
  - **Edge Threat BCE 标签二值化**：修复了边界威胁目标 `edge_threat_target` 取非二元值（0~3）直接送入 BCE 损失函数的 Bug，将其二值化为 0 或 1，消除了训练不稳定性风险。
- **运行性能加速**：
  - **持久化 Worker 加载 (`persistent_workers=True`)**：开启此选项以在 Epoch 结束时保留 DataLoader 工作进程，消除了 Windows 上每次重复创建进程的 $\sim 21$ 秒开销，且完美保留了各进程内存中的局部分局威胁缓存 `threat_grid`，极大降低了后续 Epoch 的 CPU 开销。
  - **零拷贝张量构建**：改用 `torch.from_numpy()` 代替 `torch.tensor()` 并优化数据增强对称处理时的内存连续性（`np.ascontiguousarray`），彻底消除了加载过程中的冗余内存分配与数据拷贝。
  - **非阻塞 GPU 拷贝**：采用 `non_blocking=True` 将 CPU 数据异步上传至 GPU，充分重叠 CPU 加载与 GPU 计算。
- **稳定性调整**：我们将默认的 `BATCH_SIZE` 降低为 `128`。此前使用 `256` 会在 Intel Arc 显卡驱动中触发 `UR_RESULT_ERROR_OUT_OF_RESOURCES` 显存溢出奔溃，降为 `128` 后在 B580 上能够以约 `32it/s` 极其稳定且高效地运行。

---

## 2. 训练配置与参数

主训练入口脚本为 `train.py`。它支持多个命令行参数来灵活调整训练流程。

### 命令行参数表
| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--epochs` | `int` | `50` *(见 config)* | 训练的总 Epoch 数。 |
| `--batch_size` | `int` | `128` *(见 config)* | 每个训练批次的样本数量（Intel Arc GPU 平台建议 128）。 |
| `--lr` | `float` | `1e-3` *(见 config)* | 初始学习率（采用 CosineAnnealing 调度器）。 |
| `--teacher` | `str` | `None` | 教师模型权重文件（`.pt`）的路径，用于启用**知识蒸馏**（Stage 3）。 |
| `--teacher_policy_weight` | `float` | `0.75` *(见 config)* | policy target 中 teacher soft policy 的混合权重。 |
| `--teacher_value_weight` | `float` | `0.75` *(见 config)* | value target 中 teacher value 的混合权重。 |
| `--policy_top_k` | `int` | `16` *(见 config)* | 训练前保留 top-k 合法 policy 候选点并重新归一化，用于 Top-k Policy Distillation。 |
| `--disable_top_k_policy` | `flag` | `False` | 关闭 top-k policy target shaping，保留原始 policy target。 |
| `--label_smoothing` | `float` | `0.05` *(见 config)* | 覆盖 policy label smoothing epsilon。 |
| `--entropy_alpha` | `float` | `0.01` *(见 config)* | 覆盖合法落子集合内 policy entropy regularization 权重。 |
| `--no_psq` | `flag` | `False` | 若启用此 flag，则不加载 Gomocup2017 的 `.psq` 格式数据集。 |
| `--no_npy` | `flag` | `False` | 若启用此 flag，则不加载 WinePy 的 `.npy` 格式数据集。 |
| `--seed` | `int` | `42` | 随机种子，确保实验可复现。 |
| `--resume` | `str` | `None` | 从已有 checkpoint 恢复模型、优化器和学习率调度器状态。 |

### 关键配置修改
如需修改网络深度、ResBlock 数量、输入特征平面数、不同损失的权重比例等，请编辑配置文件：
* 配置文件路径：[src/config.py](file:///C:/AAAAAAAAAAA_temp/desktop/Hephaestus_Repository/Colosseum/Gomoku_model/src/config.py)
* 在配置类 `GomokuConfig` 中，您可以调整：
  - `NUM_RES_BLOCKS`: 默认 `12`（ResNet 骨干深度）
  - `HIDDEN_CHANNELS`: 默认 `128`（特征维度）
  - `BATCH_SIZE`: 默认 `128`（以确保 Intel Arc B580 显卡训练稳定性）
  - `BATCH_COMPOSITION`: 用于解决边缘不敏感的重采样桶比例（正常、边缘落子、边缘威胁、角落落子等）
  - `ENABLE_TOP_K_POLICY` / `POLICY_TOP_K` / `POLICY_TOP_K_FLOOR`: 控制 top-k policy target shaping。
  - `TEACHER_POLICY_WEIGHT` / `TEACHER_VALUE_WEIGHT`: 控制教师模型 policy/value 蒸馏混合比例。
  - `LABEL_SMOOTHING_EPS` / `POLICY_ENTROPY_ALPHA`: 控制 policy smoothing 与合法落子 entropy 正则强度。
  - `DEVICE`: 默认为 `"cuda"`，脚本会自动检测 CUDA/XPU 架构，若不可用则会自动回退到 CPU。

---

## 3. 如何开始训练

请打开命令行并进入项目根目录：
`C:\AAAAAAAAAAA_temp\desktop\Hephaestus_Repository\Colosseum\Gomoku_model`

### 步骤 A：激活虚拟环境并确认依赖
```bash
# 激活环境
.venv\Scripts\activate

# 确认/更新依赖库
pip install -r requirements.txt
```

### 步骤 B：运行训练命令

#### 方案 1：标准多任务学习训练（推荐首次运行，从零训练完整模型）
直接运行默认配置（自动应用所有性能加速和边界增强技术），从零开始训练网络。
```bash
python train.py --epochs 50 --batch_size 128
```

#### 方案 2：轻量级快速训练（适用于配置测试/调试）
进行短期的测试训练，以便快速检查输出权重和显存占用。
```bash
python train.py --epochs 2 --batch_size 128
```

#### 方案 3：开启教师蒸馏训练（知识蒸馏模式）
如果您已经拥有一个表现较强的教师模型权重（例如 `checkpoints/best_teacher_model.pt`），希望通过大模型引导学生模型取得更好的泛化性：
```bash
python train.py --teacher checkpoints/best_teacher_model.pt --epochs 50 --batch_size 128
```

如需显式调整 Rapfi 风格的 75/25 混合比例和 top-k 候选点数量：
```bash
python train.py --teacher checkpoints/best_teacher_model.pt --teacher_policy_weight 0.75 --teacher_value_weight 0.75 --policy_top_k 16 --epochs 50 --batch_size 128
```

若要做消融实验，可关闭 top-k 或调整合法落子 entropy 正则：
```bash
python train.py --disable_top_k_policy --entropy_alpha 0.0 --label_smoothing 0.05 --epochs 10 --batch_size 128
```

#### 方案 4：分段/断点续训（增量累积训练）
如果您觉得一次性运行 50 个 Epoch 耗时过长，可以使用 `--resume` 参数加载之前保存的检查点（`checkpoints/latest_model.pt`）。脚本将自动恢复模型权重、优化器状态和学习率曲线，从上次中断的 Epoch 处继续向后训练。
```bash
# 步骤 1：首次运行，设定训练目标为 5 个 Epoch
python train.py --epochs 5 --batch_size 128

# 步骤 2：继续训练，设定新目标为 10 个 Epoch，从上一次结束的最新检查点恢复
python train.py --epochs 10 --batch_size 128 --resume checkpoints/latest_model.pt

# 步骤 3：以此类推，逐步增加目标 --epochs 并指定 --resume 即可累积完成全部 50 个 Epoch 的训练。
python train.py --epochs 50 --batch_size 128 --resume checkpoints/latest_model.pt
```

---

## 4. 训练日志与结果

### 监控训练进度
运行训练后，终端将输出 tqdm 进度条，您将能实时看到以下指标：
- `loss`: 当前批次的多任务联合损失（Total Combined Loss）。
- `pol_l`: 策略头交叉熵损失（Policy Cross Entropy Loss）。
- `val_l`: 价值头均方误差损失（Value MSE Loss）。
- `acc`: 模型的最佳落子预测准确率（以 Argmax 预测与真实落子匹配的比例计算）。

### 模型保存与权重输出
训练过程中，权重文件会自动保存在 `checkpoints/` 目录下：
1. **[latest_model.pt](file:///C:/AAAAAAAAAAA_temp/desktop/Hephaestus_Repository/Colosseum/Gomoku_model/checkpoints/latest_model.pt)**：每个 Epoch 结束时保存的最新模型状态。
2. **[best_model.pt](file:///C:/AAAAAAAAAAA_temp/desktop/Hephaestus_Repository/Colosseum/Gomoku_model/checkpoints/best_model.pt)**：在验证集上达到最低损失（Best Validation Loss）的优秀模型权重。

---

祝您训练顺利！如果有任何关于超参数调整或特定训练阶段的技术疑问，请随时告诉我。
