# 五子棋神经网络训练指南 (Gomoku Neural Network Training Guide)

本指南说明了如何启动和配置五子棋深度学习模型的训练。训练 pipeline 融合了 Rapfi 论文的网络架构（CoordConv、边界距离平面、可学习位置编码、多任务辅助头）与 D4 对称数据增强、边缘重采样等技术。

---

## 1. 代码完善度与近期修正说明

在为您准备训练说明前，我们对代码库进行了详细排查与实际运行测试：
- **缺陷修正**：发现并修复了 `src/augmentation.py` 中的一个数据增强 Bug。在对棋盘和策略分布进行 D4 对称变换（旋转/镜像）时，由于 numpy 翻转和旋转操作会返回带有**负步长（negative stride）**的视图，导致 PyTorch 在将其转换为 Tensor 时抛出 `ValueError`。我们已将 `apply_d4_symmetry` 的返回值修改为连续内存拷贝（`x.copy()` 和 `y.copy()`），消除了这一运行障碍。
- **单元测试**：成功运行了全部 18 项单元测试，结果均为 `OK`，验证了特征处理、采样桶划分、对称增强和损失函数等核心逻辑。
- **测试运行**：我们以 `batch_size=16` 运行了 1 个 Epoch 进行了短暂的训练压力测试，网络已能流畅读入数据集，成功进行前向传播、多任务损失计算、反向传播及参数更新（已在验证无误后主动中止，等待您开启正式训练）。

---

## 2. 训练配置与参数

主训练入口脚本为 `train.py`。它支持多个命令行参数来灵活调整训练流程。

### 命令行参数表
| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `--epochs` | `int` | `50` *(见 config)* | 训练的总 Epoch 数。 |
| `--batch_size` | `int` | `256` *(见 config)* | 每个训练批次的样本数量。 |
| `--lr` | `float` | `1e-3` *(见 config)* | 初始学习率（采用 CosineAnnealing 调度器）。 |
| `--teacher` | `str` | `None` | 教师模型权重文件（`.pt`）的路径，用于启用**知识蒸馏**（Stage 3）。 |
| `--no_psq` | `flag` | `False` | 若启用此 flag，则不加载 Gomocup2017 的 `.psq` 格式数据集。 |
| `--no_npy` | `flag` | `False` | 若启用此 flag，则不加载 WinePy 的 `.npy` 格式数据集。 |
| `--seed` | `int` | `42` | 随机种子，确保实验可复现。 |

### 关键配置修改
如需修改网络深度、ResBlock 数量、输入特征平面数、不同损失的权重比例等，请编辑配置文件：
* 配置文件路径：[src/config.py](file:///C:/AAAAAAAAAAA_temp/desktop/Hephaestus_Repository/Colosseum/Gomoku_model/src/config.py)
* 在配置类 `GomokuConfig` 中，您可以调整：
  - `NUM_RES_BLOCKS`: 默认 `12`（ResNet 骨干深度）
  - `HIDDEN_CHANNELS`: 默认 `128`（特征维度）
  - `BATCH_COMPOSITION`: 用于解决边缘不敏感的重采样桶比例（正常、边缘落子、边缘威胁、角落落子等）
  - `DEVICE`: 默认为 `"cuda"`，脚本会自动检测 CUDA，若不可用则会自动回退到 CPU。

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

#### 方案 1：标准多任务学习训练（推荐首次运行）
直接运行默认配置，它将结合 WinePy 自对弈数据与 Gomocup2017 比赛数据集，启用边界增强与多任务辅助头。
```bash
python train.py --epochs 50 --batch_size 256
```

#### 方案 2：轻量级快速训练（适用于配置测试/调试）
只进行 5 个 Epoch 训练，并减小 Batch 尺寸，以便快速检查输出权重和显存占用。
```bash
python train.py --epochs 5 --batch_size 64
```

#### 方案 3：开启教师蒸馏训练（阶段三）
如果您已经拥有一个表现较强的教师模型权重（例如 `checkpoints/best_teacher_model.pt`），可以使用以下命令引导学生模型进行蒸馏学习：
```bash
python train.py --teacher checkpoints/best_teacher_model.pt --epochs 50 --batch_size 256
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
