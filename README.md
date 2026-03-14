<h1 align="center">⚡ LightningMind</h1>
<p align="center">
  <b>从零手写的轻量级大语言模型全生命周期训练框架</b><br>
  <i>Pre-train ➔ SFT ➔ DPO/PPO/GRPO ➔ MoE 架构</i>
</p>

## 📖 项目简介

**LightningMind** 是一个从零手写实现的轻量级 Transformer Decoder 大语言模型。本项目不仅实现了稠密模型（Dense）的完整数据流，还从零构建了基于路由机制的**混合专家模型 (MoE)**。

更重要的是，本项目完整工程化了当前大模型最核心的训练管线，包含：
1. **预训练 (Pre-training)**：构建模型的基础语言能力与世界知识。
2. **监督微调 (SFT)**：对齐人类对话指令格式。
3. **强化学习与人类反馈对齐 (RLHF)**：全面实现了 **PPO**、**DPO** 以及 DeepSeek 提出的高效强化学习算法 **GRPO**。

---

## 🧠 核心架构与数学原理

### 1. Transformer 数据流与维度追踪 (Shape Tracking)
在底层算子实现上，本项目极其注重张量维度的精准控制与显存对齐。以核心的 `Self-Attention` 模块为例，我们在计算时严格执行因果掩码（Causal Mask），彻底杜绝未来信息穿越：

```python
# [Batch, Num_Heads, Seq_Len, Head_Dim] @ [Batch, Num_Heads, Head_Dim, Seq_Len] 
# -> Attention_Scores: [Batch, Num_Heads, Seq_Len, Seq_Len]
```

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

*注：其中 $M$ 为下三角掩码矩阵，右上角元素为 $-\infty$，确保第 $t$ 个 Token 只能与 $\le t$ 的历史 Token 发生注意力交互。*



### 混合专家架构 (MoE) 与负载均衡
为提升模型容量且不显著增加推理计算量，本项目实现了 Sparse MoE 层。为了解决训练早期的路由坍塌 (Routing Collapse)——即所有 Token 都倾向于涌入极少数专家导致算力闲置，我们在前向传播中引入了负载均衡损失 (Load Balancing Loss)：

$$L_{bal} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i$$

* $N$: 专家总数 (Number of Experts)
* $f_i$: 当前 Batch 内路由到专家 $i$ 的 Token 比例
* $P_i$: 路由网络输出给专家 $i$ 的平均门控概率
* 通过将 $L_{bal}$ 加回主干 Loss，强制要求门控网络均匀派发 Token。

# 🚀 LightningMind - 安装指南

本项目使用 [uv](https://docs.astral.sh/uv/) 作为包管理器，以确保依赖项的高效管理和环境的一致性。

## 1. 环境准备

* **Python 版本**：需要 Python **3.14** 或更高版本。

* **包管理器**：建议先安装 `uv`。如果尚未安装，请参考 [uv 官方安装指南](https://docs.astral.sh/uv/getting-started/installation/)。

  ```
  # 克隆仓库
  git clone git@github.com:snowstorm-lightning/lightningmind.git
  cd lightningmind
  
  # 使用 uv 极速创建虚拟环境并同步所有依赖
  uv sync
  ```

### 2. 数据集准备

请将相关数据集下载到本地的 `dataset` 文件夹中。 🔗 **下载地址**: [MiniMind Dataset (ModelScope)](https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files)

> 💡 **最佳实践**：默认推荐下载 `pretrain_hq.jsonl` + `sft_mini_512.jsonl`，这是最快速度复现 Zero 聊天模型的组合。数据文件可自由选择，可根据自己手头的训练需求和 GPU 资源进行适当组合。



## 🛠️ 全生命周期训练管线 (Training Pipeline)

进入训练目录：

```
cd trainer
```

### Phase 1: 预训练 (Pre-training)

构建模型的 Next-Token Prediction 能力。

```bash
# 训练稠密模型 (Dense)
uv run train_pretrain.py

# 🚀 开启 MoE 架构训练
uv run train_pretrain.py --use_moe 1
```

*(注：训练其他模型阶段的 MoE 参数使用方式同理)*

### Phase 2: 监督微调 (Supervised Fine-Tuning)

`from_weight` 表示使用哪个权重进行微调。此时要求你之前已经训练过预训练模型并生成了权重。

```bash
# 从预训练权重启动 SFT
uv run train_full_sft.py --from_weight pretrain

# 使用专家模型 (MoE) 进行 SFT
uv run train_full_sft.py --from_weight pretrain --use_moe 1
```

### Phase 3: 强化学习对齐 (RLHF: PPO / GRPO)

注意：训练 PPO 和 GRPO 模型时，需要提前下载 Reward 模型。本项目默认使用 `Skywork-Reward-V2-Qwen3-1.7B`。

Bash

```
# 1. 在项目根目录下，下载 Reward 模型到指定目录 (models 文件夹与 lightningmind 同级)
hf download Skywork/Skywork-Reward-V2-Qwen3-1.7B --local-dir "../models/Skywork-Reward-1.7B"

# 2. 回到 trainer 目录并启动 PPO 强化学习训练
cd trainer
uv run train_ppo.py --reward_model_path "../../models/Skywork-Reward-1.7B"

# 3. 或者使用更节约显存的 GRPO 算法 (需根据代码实际脚本名称调整)
uv run train_grpo.py --reward_model_path "../../models/Skywork-Reward-1.7B"
```



------

## ⚖️ 工程权衡 (Engineering & Trade-offs)

在打通全链路对齐管线时，我们深刻体会到了**显存墙 (Memory Wall)** 对底层架构设计的限制，这也是本项目实现多种 RL 算法的核心动因：

- **PPO 算法**：标准 PPO 在训练时需要同时在显存中维护 4 个模型（Actor, Critic, Reference, Reward）。即便冻结 Reference 和 Reward，状态价值网络 (Critic) 依然会消耗庞大显存，且 Actor/Critic 之间的梯度反向传播极易引发 OOM。
- **DPO 算法**：巧妙地将强化学习目标转化为分类交叉熵，直接规避了 Reward 模型，只需 Actor 和 Reference，显存占用大幅减小，但强依赖成对偏好数据 (Chosen/Rejected) 的质量。
- **GRPO 算法**：借鉴了 DeepSeek-Math 的思想，通过在同一个 Prompt 下采样多个输出形成 Group，利用组内输出的平均 Reward 作为基线，完美替代了独立的 Critic 模型。**成功省去了 1 个大模型的显存开销**，极大提升了单卡环境下的 RLHF 训练可行性与效率。


-----


## 🧪 模型测试与评估 (Evaluation & Chat)

确保需要测试的模型 `.pth` 权重文件位于 `./out/` 目录下。`-weight` 后面的参数表示要使用哪一种模型权重。

```bash
# 测试 SFT 稠密模型 (或 pretrain/dpo/ppo/grpo...)
uv run eval_llm.py -weight full_sft 

# 测试对应的 MoE 模型
uv run eval_llm.py -weight full_sft --use_moe 1
```

### 💬 精彩对话示例 (Cherry-picks)

```reStructuredText
💬: 你有什么特长？
🤖: 我是一台大型语言模型，可以回答各种问题、提供信息、进行对话和回答问题。我被设计为能够理解和生成自然语言文本，包括但不限于文本、音频、视频、图像和文本等。我的目标是提供准确和有用的信息，以帮助用户更好地理解和生成文本。如果您有任何特定的问题或需要帮助，请随时告诉我，我会尽力回答。

💬: 为什么天空是蓝色的
🤖: 天空之所以呈现蓝色，主要是因为大气层对蓝色光的散射作用。当太阳光线穿过大气层时，由于散射的光线较短，蓝光就被散射得更多，所以我们看到的天空呈现出蓝色。
在日出和日落时，太阳光线需要穿过更多的大气层，导致我们看到的天空呈现出蓝色。这就是为什么日出和日落时天空呈现出橙色或红色的美。当太阳的光线穿过大气层时，会发生散射，其中蓝色光的波长更短，因此我们看到的天空会呈现出橙色或红色。
此外，日落时，大气层会散射太阳光中的散射效应，使得太阳的光线在日落时呈现出橙色或红色。这种现象也被称为日出和日落时的色彩现象，是天文学家通过观察和研究天文学的理论。

💬: 静夜思
🤖: 床前明月光，疑是地上霜。举头望明月，低头思故乡。这首诗表达了诗人对故乡的思念之情。床前明月光，疑是地上霜。

💬: 解释一下"光合作用"的基本过程
🤖: "光合作用"是一项研究，旨在研究植物的生长和利用与其他植物、昆虫等生物之间存在的相互作用。光合作用是指植物通过吸收太阳光能将二氧化碳和水转化为有机物和氧气的过程。这些反应包括光合作用、水分解、光合作用、化学反应等。
光合作用的原理是将二氧化碳和水转化为有机物质，这些有机物质通过叶绿体中的叶绿体、根系和茎等结构结合形成氧气。这个过程涉及到叶绿素的吸收和利用，并通过光合作用产生有机物和氧气。这个过程被称为光合作用，是植物进行光合作用的主要手段。
光合作用是植物进行生命活动和繁殖的一种重要方式，它通过吸收太阳光能将二氧化碳和水转化为有机物和氧气。光合作用在光合作用中扮演着至关重要的角色，它不仅为植物提供食物和氧气，还为其他生物提供能量和养分。

💬: 如果明天下雨，我应该如何出门
🤖: 如果您明天下雨，您可以选择出门。因为天气预报显示会下雨，您应该考虑带上雨伞或者携带雨具，以防雨水淋湿衣物。此外，您也可以考虑带上一件雨伞，以防下雨。如果明天下雨，您应该穿上雨衣，以防雨水淋湿衣物。

💬: 手机和电脑的区别
🤖: 手机和电脑的区别在于它们的功能和用途。手机通常用于通讯、娱乐、娱乐、工作等。而电脑则更大型，主要用于游戏、游戏、教育等领域。此外，它们也需要使用键盘、鼠标等功能，以实现更高的效率。因此，手机和电脑的区别在于它们使用的软件和功能。

💬: 解释什么是机器学习
🤖: 机器学习是人工智能的一个分支，其目的是通过计算机程序来实现自动化分析和决策，从而使计算机能够从数据中学习。简而言之，机器学习是人工智能的一个分支，它通过让计算机自动学习并改进算法，以实现自动化决策和任务执行。

💬: 推荐一些中国的美食
🤖: 好的，中国的美食种类繁多，例如北京烤鸭、清蒸鲈鱼、豆汁、糖醋排骨、糖醋排骨等。您想了解哪些方面的美食呢？
```
