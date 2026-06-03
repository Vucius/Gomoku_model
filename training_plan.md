# 五子棋神经网络训练方案（综合改进版）

## 工程落地进度

- 2026-06-02：已落地阶段一 1.1 的输入特征平面扩展，新增 `gomoku_model/features.py`，提供当前玩家棋子、对手棋子、合法落子、边界距离、X/Y CoordConv、中心距离、步数阶段共 8 个通道。
- 2026-06-02：已新增 `tests/test_features.py`，覆盖边界距离归一化、CoordConv 坐标、合法落子、中心距离、phase 通道和当前玩家视角编码。
- 2026-06-02：已落地阶段二 2.2 的边缘局面重采样基础，新增 `gomoku_model/sampling.py`，支持按下一手落点分为 `edge_0 / edge_1 / edge_2 / center` 并按权重抽取 batch。
- 2026-06-02：已落地阶段三 3.1/3.2 的 policy label smoothing 与 visit-count temperature 软化工具，新增 `gomoku_model/targets.py`，支持 one-hot 标签、合法点均匀分布、合法点 mask 归一化、平滑标签和访问次数软化。
- 2026-06-02：已新增 `tests/test_features.py`、`tests/test_sampling.py` 和 `tests/test_targets.py`，覆盖特征编码、真实棋盘视角、边缘分桶、均衡采样、full-board 数据集 batch 编码和 policy target 变换。
- 2026-06-02：已新增 `.github/workflows/ci.yml`，在 GitHub Actions 中安装 numpy 并运行 `python -m unittest discover -s tests`；已新增 `.gitignore`，避免 `.venv` 与 Python 缓存进入版本控制。
- 2026-06-03：已落地阶段三 3.3/3.4/3.5 的训练标签控制：`gomoku_model/targets.py` 新增 top-k policy distillation、policy target 混合和 value target 混合工具；`src/trainer.py` 在训练/验证 loss 前执行 torch 版 top-k policy shaping，并将 teacher policy/value 权重拆分；`src/loss.py` 的 entropy 正则改为只在合法落子集合上计算；`train.py` 新增 `--policy_top_k`、`--disable_top_k_policy`、`--teacher_policy_weight`、`--teacher_value_weight`、`--label_smoothing` 和 `--entropy_alpha`。

下一步优先推进：把 `gomoku_model/features/sampling/targets` 与当前 `src/dataset.py` / `src/trainer.py` 做进一步去重整合，或推进阶段四的 model pool / 历史 checkpoint 对弈池。

> 基于 tmp_D1.md 与 tmp_D2.md 的分析，结合 Rapfi/MixNet 论文思路整理。

---

## 问题诊断摘要

当前训练体系存在四类核心缺陷：

| 问题 | 根因 |
|------|------|
| 边缘不敏感 | 网络缺乏空间位置感知；边缘战术样本稀缺 |
| 策略分布塌缩（One-Hot） | MCTS 后期将概率推向极端；缺乏探索多样性 |
| 自我对弈过拟合 | 数据分布越来越窄；单一来源的训练闭环 |
| 搜索/训练解耦不足 | 搜索质量决定标签质量；两者互相阻塞 |

---

## 阶段一：网络结构改进（解决边缘不敏感）

### 1.1 输入特征平面扩展

在原有的"当前玩家棋子平面 + 对手棋子平面"基础上，**必须**额外拼接以下通道：

```
通道 1：当前玩家棋子平面
通道 2：对手棋子平面
通道 3：合法落子平面
通道 4：边界距离平面  ← 最关键
通道 5：X 坐标归一化平面（CoordConv）
通道 6：Y 坐标归一化平面（CoordConv）
通道 7：中心距离平面（可选）
通道 8：当前步数/阶段平面（可选）
```

**边界距离平面计算方式：**
```python
edge_distance[i][j] = min(i, j, H-1-i, W-1-j)
# 归一化到 [0, 1]
edge_distance = edge_distance / edge_distance.max()
```

这使网络能明确感知"当前点在角落/边缘/二线/中心区域"，而不是强加固定先验。

### 1.2 位置编码选型

- **不推荐**：固定三维高斯负 bias（硬编码先验，破坏平移不变性）
- **推荐方案 A**：CoordConv 坐标通道（见 1.1，数学确定性更高）
- **推荐方案 B**：可学习 Position Embedding
  ```
  PositionEmbedding ∈ R^(C × H × W)
  或分解为：row_embedding[i] + col_embedding[j]
  ```
- **高级方案**：参考 Rapfi 的 Pattern Codebook——将棋盘分解为水平、垂直、主对角、副对角四个方向的长度为 11 的一维线条模式（Line Patterns），边缘截断局面作为独立显式状态录入 Codebook，查表即可无损识别边界，完全不依赖卷积"猜测"边界。

### 1.3 辅助任务头（多任务学习）

训练时加入辅助预测头，推理时可丢弃：

```
threat_type_head    ：预测每个点是否形成活三/冲四/活四
distance_to_win_head：预测几手内是否存在强制胜
edge_threat_head    ：专门预测边缘附近威胁点  ← 针对边缘问题
legal_move_head     ：预测合法点
ownership_head      ：预测区域影响力（可选）
```

辅助头帮助网络学习"为什么这个点重要"，而不仅仅是"最佳点是哪里"。

---

## 阶段二：数据来源多样化（解决过拟合与分布塌缩）

### 2.1 实际训练数据池构成

针对当前项目的具体情况，我们已将数据池调整为以下三类主要来源：

1. **WinePy 自我对弈数据集 (`dataset/gomoku_dataset_split/`)**
   - 规模：875 局完整对弈，共 26,378 个落子样本。
   - 作用：提供基础自对弈样本，评估模型基本性能与收敛情况。
2. **Gomocup 2017 比赛数据集 (`dataset/gomocup2017/`)**
   - 规模：共计 **25,848 个** 真实引擎对弈 `.psq` 游戏文件。
   - 构成：`fastgame` (15,984 局), `freestyle` (8,280 局), `renju` (504 局), `standard` (1,080 局)。
   - 作用：提供高质量的强引擎竞技棋谱，涵盖多种游戏规则与开局变化。
3. **Karesis/Gomoku Hugging Face 在线数据集 (通过 `datasets` 库载入)**
   - 作用：在线引入数十万高质量对弈局面，彻底打破自我对弈数据过拟合问题。
   - 增强：对每个样本进行 D4 群 8 种镜像与旋转变换，直接将数据量扩增 8 倍。

*(注：经盘点，`gomocup2023results` 文件在本地并不存在，故未予集成。)*

### 2.2 边缘局面专项数据再平衡

按落点距边界距离分桶：

```
角落区域（距离边界 0）：保证每 batch 有一定比例
边缘区域（距离边界 1）：同上
近边区域（距离边界 2）：同上
中心区域（距离边界 ≥ 3）：主体
```

**推荐 batch 构成：**
```
40%：正常自我对弈样本
20%：边缘落子样本
20%：边缘威胁样本
10%：角落附近攻防样本
10%：人工构造战术样本
```

**边缘战术样本需覆盖：**
```
边缘活三 / 冲四 / 跳三 / 眠三
边缘双三 / 四三
边缘冲四防守
边缘双威胁
角落附近 VCF / VCT
靠边长连限制
（Renju 规则：黑棋靠边禁手判断）
```

### 2.3 战术局面生成器

单靠自我对弈无法自然产生足够的边缘战术局面，需额外构建生成器：

```
推荐流程：
1. 随机生成靠边 Pattern（合法性检查）
2. 用强搜索器（α-β / VCF/VCT 专项搜索）搜索最佳着法
3. 保存 policy target 和 value target
4. 加入训练集（周期性更新）
```

---

## 阶段三：训练标签改进（解决策略塌缩）

### 3.1 Policy Label Smoothing

```python
π_smooth = (1 - ε) × π_mcts + ε × uniform_legal
# ε 建议范围：0.03 ~ 0.10
# 15×15 棋盘上合法点多，ε 不宜过大，或改为只对 top-k 合法点平滑
```

### 3.2 Softmax Temperature 软化

```python
π_i = visit_i^(1/τ) / Σ visit_j^(1/τ)

# 训练早期：τ = 1.0
# 训练中期：τ = 0.7
# 训练后期：τ = 0.5
# 不建议 τ 长期接近 0（会退化为 one-hot）
```

### 3.3 Top-k Policy Distillation

保留 top-k 候选点（k = 8 / 16 / 32），对 top-k 内重新归一化，其余点给极小概率。

**当前落地状态**：`gomoku_model.targets.top_k_policy()` 提供 numpy 工具；训练时 `GomokuTrainer.shape_policy_target()` 会对原始/teacher 混合后的 policy target 执行 top-k 截断，默认 `k=16`，非 top-k 合法点保留 `1e-6` floor 后重新归一化。

### 3.4 策略熵正则化

在 policy 损失函数中加入熵惩罚项：

```
L_p_total = L_p_cross_entropy - α × Σ π̂(m) × log π̂(m)
```

惩罚"过于绝对"的单点输出，迫使训练时维持一定的分布宽度。

**当前落地状态**：`src/loss.py` 已实现合法落子集合内的 policy entropy regularization，避免鼓励非法/已占点概率。

### 3.5 混合标签策略（核心）

参考 Rapfi 论文，采用混合监督：

```
policy target = 0.75 × teacher_soft_policy + 0.25 × search_policy
value target  = 0.75 × teacher_soft_value  + 0.25 × game_result

# 可调比例范围：
#   teacher : original = 75:25（论文推荐）
#   teacher : original = 50:50（折中）
#   teacher : original = 25:75（数据充足时）
```

更激进的混合（用于边缘问题）：
```
policy target = 0.5 × Gumbel MCTS policy
             + 0.3 × AlphaBeta best-line policy
             + 0.2 × teacher policy
```

**当前落地状态**：训练器已将 teacher policy 与 teacher value 权重拆分为 `TEACHER_POLICY_WEIGHT` / `TEACHER_VALUE_WEIGHT`，CLI 可分别覆盖；`gomoku_model.targets.mix_policy_targets()` 和 `mix_value_targets()` 可用于后续搜索器/教师标签离线生成。

---

## 阶段四：自我对弈多样性（避免分布塌缩）

### 4.1 Model Pool 对弈池

维护一个模型池：

```
latest model          (最新模型)
best model            (历史最佳)
previous 5 checkpoints
random older checkpoint
teacher model
rule-based / search-based agent
```

**对弈匹配比例：**
```
60%：latest vs latest
20%：latest vs historical best
10%：latest vs random old checkpoint
10%：latest vs teacher/search agent
```

### 4.2 开局多样化

```
10%：第一手在中心附近
20%：前 3 手随机落子
20%：从开局库采样
20%：从边缘/近边局面开始（专项强化）
30%：正常自我对弈
```

> 注意：五子棋中心先手优势强，完全随机第一手会产生质量较差的数据。推荐从**平衡局面库**或**人工生成的非劣局面**出发。

### 4.3 MCTS 探索增强

- 根节点 Dirichlet 噪声强度适当加大
- 搜索初期温度参数 τ 维持在较高水平更长时间
- 强制搜索树扩展边缘区域（哪怕网络认为某步是绝对好棋）

---

## 阶段五：数据增强

### 5.1 D4 对称增强（必须做）

```
旋转 90° / 180° / 270°
水平翻转
垂直翻转
主对角翻转
副对角翻转
```

直接扩大 8 倍数据量。

> **注意**：D4 增强只能让方向更均衡，无法凭空产生边缘战斗数据，需与边缘采样配合使用。

### 5.2 颜色交换增强

若输入采用"当前玩家/对手"相对平面（而非黑白绝对平面），可做颜色交换增强，需注意先后手和规则约束。

---

## 阶段六：搜索与推理优化

### 6.1 搜索和训练解耦

```
训练数据生成阶段：重搜索，追求标签质量
实战推理阶段    ：轻搜索，追求速度
```

**数据生成可用搜索器：**
```
Gumbel MCTS
Alpha-Beta / PVS
VCF/VCT 专项搜索
Proof-Number Search
Threat-Space Search
```

### 6.2 并行架构

```
Self-play workers  →  Replay Buffer  →  Trainer  →  Model Pool
       ↑                                                  ↓
       └──────────────── latest checkpoint ───────────────┘
```

**并行要点：**
```
- 自我对弈多进程并行，每个 worker 持有模型副本
- 批量请求神经网络推理（Batch Inference）
- 搜索树内部使用虚拟损失（Virtual Loss）
- Gumbel candidate expansion 并行化
- 数据写入使用 replay buffer / queue
- 训练进程和数据生成进程完全解耦，互不阻塞
```

> 注：Python 多进程受 IPC 和内存序列化瓶颈限制，推荐使用 C++ 线程池或 ONNX Runtime 并行执行器，构建全局"推理请求队列"。

### 6.3 推理加速（部署阶段）

```
整网量化：16-bit / 8-bit 整数量化
向量化  ：AVX2 / AVX-512 SIMD 指令（实测约 4× 性能提升）
增量更新：借鉴 Rapfi 的增量哈希/特征更新机制
```

---

## 完整训练路线（7 阶段）

```
阶段 1：基础监督预训练
  - 收集人类棋谱、开源棋谱、强代理棋谱
  - 使用 D4 对称增强
  - 训练 policy/value 网络基础版
  - 加入位置编码（edge distance plane + CoordConv）

阶段 2：教师模型训练
  - 训练较大的 ResNet / ConvNeXt-like 教师模型
  - 使用更强搜索器（α-β + Gumbel MCTS）生成 soft policy/value
  - 教师模型不追求推理速度，只追求标签质量

阶段 3：学生模型知识蒸馏
  - 小模型学习教师模型输出
  - 使用 75% teacher soft label + 25% 原始搜索标签
  - policy 使用 label smoothing + temperature 软化

阶段 4：边缘专项数据修复
  - 构造 edge tactical dataset（战术局面生成器）
  - 训练 batch 提高边缘样本采样权重
  - 激活 edge_threat_head 辅助任务头
  - 对边缘局面单独验证和基准测试

阶段 5：自我对弈强化迭代
  - 启用 model pool（latest / old / teacher 混合对弈）
  - 随机开局 + 平衡开局混合
  - 策略熵正则化防止 one-hot 退化
  - 保留数据多样性，不只保留最新模型数据

阶段 6：搜索增强标签回灌
  - 对关键局面使用 Gumbel MCTS + AlphaBeta + 威胁搜索重新分析
  - 将高质量分析结果回灌训练集
  - 混合 policy target（多搜索器加权融合）

阶段 7：部署模型压缩
  - 蒸馏到推理小模型
  - 量化（INT8/INT16）
  - AVX2/AVX-512 向量化加速
  - 搜索端并行优化（Batch Inference + Virtual Loss）
```

---

## 最高优先级改进点（如只能选 5 个）

| 优先级 | 改进项 | 预期效果 |
|--------|--------|----------|
| ⭐⭐⭐ | 输入加入 edge distance plane + CoordConv | 根治边缘不敏感 |
| ⭐⭐⭐ | 边缘局面 batch 重采样 + 战术生成器 | 边缘数据充足 |
| ⭐⭐⭐ | policy target 做 label smoothing + temperature 软化 | 消灭策略 one-hot |
| ⭐⭐⭐ | 引入教师模型蒸馏（75% soft label） | 缓解过拟合 + 提升标签质量 |
| ⭐⭐ | 使用 model pool + 历史版本对弈 | 避免自我对弈分布塌缩 |

---

*方案综合自 tmp_D1.md（详细分析）与 tmp_D2.md（Rapfi/MixNet 论文对照），并参考 Rapfi 论文中 Katagomo 自对弈 3080 万局面 + ResNet-6b128f 教师蒸馏 + MixNet 75/25 混合监督的工程实践。*
