# 08–11 研究手册：Embedding Geometry & High-Dimensional Retrieval

该模块研究三种不同的「维度」：环境坐标数量 `D`、潜在生成因子数量 `d`、谱有效秩 `r_eff`。三者不等价。

## 08 非线性 Swiss Roll

- **合成真值**：`(t, h)` 两个自由变量映射到 `X=(t cos t, h, t sin t)`；再用列正交的 `Q∈R^(64×3)` 映射至64维。
- **有意义的距离**：展开后的流形坐标 `(s(t),h)`，其中 `s(t)=0.5[t sqrt(1+t²)+asinh(t)]`。在内部局部区域它近似流形最短路径的几何，但边界/非凸路径可能造成偏差。
- **比较**：原64维欧氏距离、PCA-2、Isomap-2。统一使用训练集610个点拟合，170个留出点评估 `10-NN` 与真值邻域重叠率。
- **假设**：PCA主要关注方差，不强制恢复曲面上的弧长结构；Isomap使用邻接图最短路尝试恢复内在几何。
- **失效模式**：Isomap对 `n_neighbors` 和图连通性敏感，短路边会伤害展开效果；高维环境维度本身不意味着破坏信息。
- **可视化**：二维颜色轨迹展示卷曲/展开，邻域重叠柱状图给出定量证据。色彩的连续性不能替代客观指标。

## 09 各向异性、Hubness 和向量检索

构造两个严格对照世界：

1. **nuisance**：相同类别共享8维语义中心，另有均值偏移和巨大的类无关主成分（nuisance）。假设剔除 PC1 可以改善余弦检索。
2. **signal**：类别标签恰好承载在最大的主成分中。假设剔除 PC1 会伤害分类/检索。

所有处理均 **只使用 database/documents 拟合**：

- `raw`：原始余弦。
- `centered`：减去数据库均值。会改变余弦，不会改变欧氏距离。
- `remove_top_pc`：减均值后删除数据库居中数据第一主成分方向；这可能删除语义信号。
- `whiten_top_components`：数据库 PCA 的前16个主轴变换并缩放；绝非通用推荐的检索设置。

**任务指标**：`top1_accuracy`（首条标签是否匹配）、`precision@5`（前5条相关比例）、`hit@5`（前5条至少一个相关）。随机索引语义基线为 `1/num_classes`（类别平衡）。

**几何指标**：`hubness_gini` 是某文档出现在所有 query 的 Top-k 中的次数分布的 Gini；它不是所有 hubness 定义的唯一实现，也不能单凭 Gini 评价质量。

**重要注意**：此实验的「类别」只是合成标签，**不能说明真实语义**，也不是对任意生产 Embedding 算法的评测。图表与 `metrics.json` 是唯一可信的实验结果，不预设 PC 移除必然有效。

## 10 谱有效维度与邻居稳定性

模型：`X = ZQᵀ + σ ε`，其中 `Z∈R^(N×6)`，`QᵀQ=I`，`ε` 为128维同方差高斯噪声。

比较 `σ∈{0,.02,.05,.1,.2,.35,.6}` 期间的：

- Shannon 有效秩：`exp[-Σ p_i ln(p_i)]`，`p_i=λ_i/Σλ_i`。
- participation ratio：`(Σλ_i)²/Σλ_i²`。
- 在真实潜变量 Z 定义的 kNN 邻域相对下：原始128维与训练PCA-6维的邻居重叠率。

**坑点**：噪声只要全秩，谱有效秩就可能上升，但产生机制仍只有6个语义潜变量；有效秩在噪声下不总是「真实内在维度」。

## 11 可选 PyTorch Autoencoder

使用64维无损线性嵌入 Swiss Roll，对比：PCA-2、Isomap-2、非线性 Autoencoder（64→64→32→2→32→64→64）。AE 训练仅看640个训练样本的**重构损失**，不访问 oracle `s(t),h`；测试集仅用于评价，并画出训练过程的留出重构误差。

两个独立评价目标：

- `heldout_reconstruction_mse`：重构向量是否接近原向量。
- `oracle_neighbor_overlap_at_10`：学得的潜变量是否保持真实流形上的邻近关系。

AE 很可能找到低重构误差却未实现展开的潜变量；这是有效反例，而不是 bug。修改隐藏层宽度、优化步数、噪声和训练样本数量，理解归纳偏置。

## 生产向量的离线审计

构造 `.npz`：

```python
import numpy as np
np.savez_compressed('my_vectors.npz',
    doc_embeddings=database_vectors,   # [N,D] float
    query_embeddings=query_vectors,    # [M,D] float；与 database 分开
    doc_labels=database_category_ids,  # [N] 可选
    query_labels=query_category_ids,   # [M] 可选
)
```

运行 `curse-of-dim-audit --input my_vectors.npz --out outputs/audit --k 5`，输出 `audit.json` 和 `audit.png`。可独立读取有效秩、Top-1/Precision@K/Hit@K、Hubness Gini。**没有标签就不声称语义质量**。真实语义搜索建议用 relevance judgments (`query_id`, `doc_id`, score) 对应到 MAP/nDCG，未来版本可加入。

## 研究文献

- Ethayarajh (2019), *How Contextual are Contextualized Word Representations?* https://aclanthology.org/D19-1006/
- Radovanović, Nanopoulos & Ivanović (2010), *Hubs in Space*. https://www.jmlr.org/papers/v11/radovanovic10a.html
- Tenenbaum et al. (2000), *A Global Geometric Framework for Nonlinear Dimensionality Reduction* (Isomap). https://doi.org/10.1126/science.290.5500.2319

所有实验都是确定种子的合成模拟，不等价于这些论文的严格复现。真实数据域偏移、相似度校准、负例选取会显著改变结果。
