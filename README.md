# Curse-of-Dim · 维度灾难实验室

[![Python CI](https://github.com/1998x-stack/Curse-of-Dim/actions/workflows/ci.yml/badge.svg)](https://github.com/1998x-stack/Curse-of-Dim/actions/workflows/ci.yml)

**7 个可复现、具备数学预测和严格对照的实验：从高维几何、距离集中、KNN 退化、样本复杂度，到内在维度、Double Descent 和优化病态。**

> 核心问题不是「维度越大就越差」，而是「数据是否具有可利用的低维结构和任务相关信号，以及算法能否发现它」。

## 快速开始

推荐 Python 3.10+。在 Linux / macOS / Windows (PowerShell) 均可运行。

```bash
git clone https://github.com/1998x-stack/Curse-of-Dim.git
cd Curse-of-Dim
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev,notebook]'
python -m curse_of_dim --list
python -m curse_of_dim --experiment all --out outputs
python -m pytest -q
```

`curse-of-dim --experiment geometry,distances --out outputs/part1` 运行部分实验，也可以重复指定 `-e geometry -e distances`。输出 `.png` 和 `metrics.json`，不会把随机生成图覆盖到源代码目录。

## 实验与结果图

| ID | 研究问题 | 输出 |
|---|---|---|
| 01 `geometry` | 固定分辨率下覆盖空间需要多少网格？ | `01_geometry.png` |
| 02 `distances` | 为什么距离相对波动随维度变小？ | `02_distances.png` |
| 03 `knn_noise` | 无关维度为何使 KNN 的近邻排序变差？ | `03_knn_accuracy.png`, `03_knn_slices.png` |
| 04 `sample_complexity` | 样本量增加如何影响局部非参数拟合？ | `04_sample_complexity.png` |
| 05 `intrinsic_dimension` | 128维表示只有2维有效信息时会怎样？ | `05_intrinsic_dimension.png` |
| 06 `double_descent` | 插值阈值为何可能出现测试误差峰值？ | `06_double_descent.png` |
| 07 `conditioning` | 只有2维，为什么优化仍可能极慢？ | `07_ill_conditioning.png` |

**深度阅读：** [docs/EXPERIMENT_GUIDE.md](docs/EXPERIMENT_GUIDE.md) 中逐一解释假设、公式、控制变量、可视化意义、局限与反例。

**交互 Notebook：** [notebooks/curse_of_dimensionality.ipynb](notebooks/curse_of_dimensionality.ipynb)，另有可独立阅读的中文实验教材。启动：`jupyter lab notebooks/curse_of_dimensionality.ipynb`。

## 工程布局

```text
Curse-of-Dim/
├── .github/workflows/ci.yml     # 多 Python 版本 CI
├── docs/EXPERIMENT_GUIDE.md     # 数学推导与因果对照
├── notebooks/curse_of_dimensionality.ipynb
├── src/curse_of_dim/
│   ├── __main__.py              # python -m curse_of_dim
│   ├── cli.py                   # 可选实验 CLI 和指标 JSON
│   ├── experiments.py           # 7 个自包含实验及绘图
│   └── theory.py                # 解析公式和几何基准
├── tests/                       # 数学性质、CLI、绘图烟雾测试
├── pyproject.toml               # pip 安装 + console script
└── Makefile
```

## 可复现性和科学边界

- 各实验使用固定 `np.random.default_rng(seed)`，不依赖 notebook 的全局随机状态。
- 实验图采用非交互式 `Agg` 后端，可在无桌面的服务器/CI 生成。
- 测试会检查重要数学恒等式、严格等距嵌入及距离集中趋势。
- `geometry` 的最近邻 `L∞` 半径在**周期单位立方体**上推导，不能直接套到普通欧氏空间。
- `distances` 的集中结论限定为**独立均匀坐标**，并非任意高维表示。
- `double_descent` 的性质只对应给定随机线性模型，不意味着所有深度模型都呈双降。
- 大部分运行时间来自 PCA、KNN 和最小范数回归；CPU 即可运行，无 GPU 要求。

## License

MIT, see [LICENSE](LICENSE).

## 深入扩展（08–11）：流形、Embedding 与检索

新实验详见 [docs/EMBEDDING_GEOMETRY.md](docs/EMBEDDING_GEOMETRY.md) 和 [可交互深入实验 Notebook](notebooks/embedding_geometry_deep_dive.ipynb)。均可使用 CPU，无需下载真实模型或外部数据。

| ID | 问题 | 图 |
|---|---|---|
| `nonlinear_manifold` | Swiss roll：线性 PCA 能否正确展开非线性流形？对比 Isomap 的邻居保持率 | `08_nonlinear_manifold.png` |
| `embedding_anisotropy` | 均值方向、第一主成分和 cosine 检索；包括移除 PC1 使结果变差的反例 | `09_anisotropy_retrieval.png` |
| `effective_dimension` | 128维表示中仅6个潜在因素；噪声如何改变有效秩和近邻重叠率 | `10_effective_dimension.png` |
| `autoencoder_manifold` | **可选 PyTorch**：比较2维瓶颈 AE、PCA、Isomap；检验重构误差与邻域恢复的差异 | `11_autoencoder_manifold.png` |

```bash
# 默认全部实验 01–10（无需 PyTorch）
curse-of-dim -e all --out outputs/all

# 仅高级实验
curse-of-dim -e nonlinear_manifold,embedding_anisotropy,effective_dimension --out outputs/deep

# 可选深度学习实验（单独安装 PyTorch）
pip install -e '.[deep]'
curse-of-dim -e autoencoder_manifold --out outputs/autoencoder

# 真实 Embedding 本地审计（不会下载、上传或泄漏你的向量）
curse-of-dim-audit --input /path/to/my_vectors.npz --out outputs/my_audit --k 5
```

真实向量文件 NPZ 必须含 `doc_embeddings` 与 `query_embeddings` 两个二维数值数组；可选提供 `doc_labels` 与 `query_labels` 用于 `precision@k` 和 `hit@k`。查询集合必须独立于数据库集合；否则检索结果存在身份泄漏。所有标准化及 PCA 都仅对数据库 `fit`，然后应用于查询。

> 注意：有效秩不是严格的流形内在维度，均值中心化只能改变余弦几何、无法改变欧氏距离，删除 PC1 和白化都不是通用性能改进策略。使用真实 Embedding 时必须检查任务指标，而不能仅根据各向同性程度下结论。
