# ZeroDiff 类别关系保持诊断

本实验只检验：语义空间与真实视觉类别中心共同支持的远近排序，是否在原始 ZeroDiff 正式生成后发生反转。不训练模型，不使用 PaCo、RSC、SDGA、GSR，不比较扩散状态。

## 服务器准备

按项目现有的 Git 流程提交并同步本次新增文件，在服务器项目根目录使用原有 ZeroDiff 训练环境运行。无需上传或解压运行包，也无需同步 `log/`、`out/` 中的权重。脚本直接复用项目中的基线网络定义和正常推理接口，不需要导入训练脚本。

需要以下文件，均可留在服务器，无须下载：

- `Dataset/AWA2/res101.mat`：原始样本标签。
- `Dataset/AWA2/att_splits.mat`：类别属性、名称及划分。
- `Dataset/AWA2/ce_ce.mat`：与该 checkpoint 一致的 2048 维视觉特征。
- 纯 ZeroDiff DFG checkpoint，以及该次训练实际使用的 DRG checkpoint。
- 实际训练配置 JSON，推荐明确提供；缺省时查找 DFG 同目录的 `config.json`，再缺省则沿用仓库 AWA2 默认配置，并在 manifest 中记录。

`docs/feature_export_awa2.example.json` 仅为配置示例，不能未经核对当作训练记录。若模型使用 10% 或 30% 训练数据，须提供相应 split MAT，并设置 `--split-percent`；首轮建议使用 100% 模型。

确认正常评估用 `one-step` 还是 `progressive`，使用相同参数，不能根据诊断反转率选采样方式。模型文件名无法可靠区分这两种方式。

依赖沿用项目环境：Python、PyTorch、NumPy、SciPy、scikit-learn；图表直接输出 SVG，不需要安装 Matplotlib。建议 Python 3.10 或更新版本。服务器不需要联网。

## 运行

在项目根目录运行，替换下面的路径。以下示例中的 `one-step` 必须核对实际评估设置后使用。

```bash
python scripts/diagnose_awa2_relations.py \
  --dataroot /absolute/path/to/Dataset \
  --dfg-checkpoint /absolute/path/to/DFG.tar \
  --drg-checkpoint /absolute/path/to/MATCHING_DRG.tar \
  --training-config /absolute/path/to/config.json \
  --split-percent 100 \
  --sampling one-step \
  --device cuda:0 \
  --output-dir out/relation_diagnostic/AWA2_run1
```

可先在同一命令末尾加 `--check-inputs`，只检查路径。正式运行保留原命令、去掉该参数。输入目录为包含 `AWA2/` 的父目录。输出目录已存在时会拒绝覆盖；重新运行请换目录。

默认设置：10 个未见类，每类 100 个生成特征，5 个生成随机种子（9182–9186），每类 batch size 100，语义及真实视觉相对距离间隔阈值 0.05，生成并列阈值 0.000001。

## 指标约定

先平均视觉特征得到类别中心，再计算余弦距离；类别语义直接计算余弦距离。不进行 PCA、t-SNE 或可学习投影。

对每个锚点类别，遍历其余类别的无序对，并按语义距离确定 near/far。10 类共 360 个候选三元组。定义相对间隔：

```text
gap = (distance_far - distance_near) / max(distance_far + distance_near, 1e-12)
```

仅保留语义 gap > 0.05 且真实视觉 gap > 0.05 的三元组，筛选不看生成结果。五次生成共享同一有效集合。

- 生成 gap < -0.000001：严格反转。
- 生成 |gap| <= 0.000001：并列，单独统计，不算严格反转。
- 生成 gap > 0.000001：保留。

主指标是有效集合上的严格反转比例，报告五次采样的均值、样本标准差与范围，另报并列率和有效集合覆盖率。无有效三元组时指标是未定义，不报告为零。零向量无法定义余弦距离，脚本报错。

展示案例优先来自超过半数采样出现反转的候选，按反转频次距中位数最近选择，再按固定遍历顺序打破并列。若没有多数重复的反转，展示观察到的案例并明确次数；没有反转则不生成反转案例。显示的种子取该案例反转间隔的中位数附近。案例不是总体证据，总体频率须同时展示。

## 返回文件

只需将输出目录中的 **`results.zip`** 下载回来。包内包含：

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 主指标、覆盖率、案例索引和结果状态 |
| `per_seed.csv` | 每次生成的反转数、比例与并列率 |
| `triplets.csv` | 全部候选及各次生成距离、筛选标记、类别名称，可审计 |
| `centers.npz` | 语义、真实及生成中心和距离矩阵，可离线复算 |
| `figure1.svg` | 实测案例与五次采样反转率，独立矢量图 |
| `report.md` | 简明实验结果及解释边界 |
| `manifest.json` | 数据、权重、配置和源码哈希，参数及环境信息 |

原始图表用英文标签，便于后续排入论文；案例内三种空间的距离条分别缩放，实际值标在旁边，不能跨空间比较条长。

若观察到可重复的反转，可支持“该 ZeroDiff checkpoint 生成的类别中心未总能保持真实视觉也支持的语义排序”。若未观察到，应如实报告；本实验不推导类内失真、具体原因或任何状态权重方案的必要性。五个生成种子不等同于五个独立训练模型。

## 软件验证

```bash
python -m unittest discover -s tests -p test_relations.py -v
```

软件测试采用已知几何或构造数据，不构成 ZeroDiff 的实测证据。
