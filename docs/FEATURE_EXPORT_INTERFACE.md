# ZeroDiff AWA2：特征导出与离线绘图接口

接口版本：`awa2-visual-features-v1`。

本项目导出 ZeroDiff baseline，Rediff 在其项目中分别导出；两边沿用同一份接口和比较
协议，独立可视化项目只读取 NPZ 和 JSON。本次适配的范围是 ZeroDiff 导出端，
可视化项目单独建立。导出程序不训练分类器、不更新权重、不执行 PCA/t-SNE，也不依赖
`diagnostics` 或 `relation` 包。范围固定为 AWA2 的全部 10 个未见类、2048 维视觉
特征 V。`gzsl_VCS` checkpoint 也只导出其 DFG 产生的 V，不是 VCS 拼接特征。

## 1. 两张参考图分别用什么

| 模式 | 文件 | 用途 |
| --- | --- | --- |
| `generated-only` | `generated_only.npz` | 第一张：不同方法生成的未见类特征，各类使用固定颜色 |
| `real-vs-generated` | `real_vs_generated.npz` | 第二张：同类真实与生成特征，空心圆表示真实，星号表示生成 |
| `both`（默认） | 两个文件 | 只生成一批特征；两个文件的生成特征、标签和样本标识完全相同 |

默认每类生成 100 个样本，共 1000 个；真实特征保留全部 `test_unseen_loc` 样本。
第一张参考图没有给出样本数量，因此这里是统一实验设置，不声称还原其精确数量。
生成样本是从指定 checkpoint 重新采样，不是恢复当年训练评估时的那批随机特征。

“方法 × 100%/30%/10% 训练样本”的六面板，需要两个项目各自加载三种训练比例的
checkpoint，共导出六组。更改 `--split-percent` 只是声明并核对训练条件，不会把
100% 模型变成 10% 模型。

## 2. 输入准备与训练配置

在各自训练项目根目录运行，需要现有训练环境中的 PyTorch、NumPy、SciPy、
scikit-learn。没有新增第三方依赖。数据布局：

```text
Dataset/AWA2/
  res101.mat          # 原始样本标签（导出不读取其中的视觉特征）
  att_splits.mat      # trainval/test_unseen 划分、属性、类别名称
  ce_ce.mat           # V 特征，行数为总样本数，列数为 2048
  split_30percent.mat # 导出 30% 条件时要求存在，记录其哈希
  split_10percent.mat # 导出 10% 条件时要求存在，记录其哈希
```

不需要加载 `con_paco.mat`：生成时的对比表征来自 DRG，真实对照只使用 V。
真实特征的预处理严格沿用 `datasets/image_util.py`：先在完整 `trainval_loc` 上拟合
MinMaxScaler（或配置指定的 StandardScaler），转成 float32 后除以训练特征最大值。
百分比抽样在原训练流程中发生于这一步之后，因此不在 10%/30% 子集上重新拟合。
不会对未见类拟合 scaler，也不会在导出后单独归一化真实或生成特征。

必须提供 DFG checkpoint 及**该次训练实际使用的 DRG checkpoint**。DFG 键为
`state_dict_G`；DRG 支持 `state_dict_G_con` 或 `state_dict_R`。
baseline 和带 `state_dict_VSRA` 的方法均支持；VSRA 是训练模块，生成路径无需加载它。
推荐使用论文结果对应的 best-model 权重文件，而不是 `*_training_last.tar` 恢复文件。

程序优先读取 `--training-config`；未提供时自动查找 DFG 同目录的 `config.json`。
支持 `{"training": {...}}` 格式，也支持直接放置配置字段的 JSON 对象，包括带 UTF-8
BOM 的 Windows 文件。当前 ZeroDiff 训练脚本不会自动生成该 JSON，需依据实际训练参数
准备；可参考 `docs/feature_export_awa2.example.json`，修改 `split_percent` 和有差异的设置。
存在 `split_percent` 时必须与命令一致；存在 `drg_sha256` 时必须与实际 DRG 权重一致。
`drg_sha256` 可选，填写时必须为 64 位小写十六进制摘要。布尔配置必须写 JSON 的
`true` / `false`，不能写字符串 `"false"`；不完整配置使用的默认字段会输出并记入 manifest。

例如，先复制示例并核对参数，再计算实际配套 DRG 的 SHA256，将其添加到 `training` 对象：

```bash
python -c "import hashlib; from pathlib import Path; p=Path('PATH_TO_MATCHING_DRG.tar'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

旧版 ZeroDiff 长文件名模型可能没有配置。此时程序会明确输出提示，使用与当前仓库
`scripts/run_awa2_zerodiff_DFG_train.py` 对齐的 AWA2 推理默认值，并在 manifest 标记为
`published_defaults_user_declared`（保留原接口字段值）：

| 配置 | 默认值 |
| --- | --- |
| `resSize / attSize / noiseSize / dim_t` | `2048 / 85 / 85 / 85` |
| `decoder_layer_sizes`（有效值） | `[4096, 2048]` |
| `n_T / ddpmbeta1 / ddpmbeta2` | `4 / 0.1 / 20.0` |
| `class_embedding / class_embedding_norm` | `att / true` |
| `preprocessing / standardization / validation` | `true / false / false` |

若实际训练设置不同，应提供真实配置，尤其是权重形状无法反映的扩散步数、beta 和
预处理设置。`decoder_layer_sizes` 最后一项会像原训练参数解析器一样重写为 2048。
没有配置时，工具不能独立证明模型训练比例或 DRG 配对是否正确，这些是调用者的声明。

`--sampling` 必须显式选择 `one-step` 或 `progressive`。原评估会把两种采样得到的
最佳模型写到相同的 `gzsl_VCS` 路径，不能凭文件名判断使用哪种采样。比较组应采用
预先确定的相同规则；该参数会写入比较协议。导出算法与原 `sample_from_model` 对齐：
类别属性 → 随机 DRG 对比特征 → 单步 DFG 或逆扩散 DFG，不使用真实未见类特征作为
生成条件，也不使用编码器重建真实样本。

## 3. 在当前 ZeroDiff 项目运行

下面是 Bash 多行命令；Windows PowerShell 可合并为一行执行，或将换行符改为反引号。
替换示例中的权重路径。默认输出到项目根目录下的
`out/plot_features/AWA2/<训练比例>pct/`：100%、30%、10% 分别使用 `100pct`、
`30pct`、`10pct`。`--output-dir` 可省略，也可显式指定其他目录；`out/` 已被 Git 忽略。

### 三种训练比例一起导出

`scripts/export_awa2_plot_features.py` 已按当前 AWA2 权重列表填写三组
`DFG *gzsl_VCS.tar` 和 `DRG *_gzsl.tar` 的完整文件名，默认依次导出 100%、30%、10%。
DRG 的选择依据仓库训练命令；若实际训练使用了不同的 `--netR_model_path`，应先修改
脚本的 `checkpoint_paths`，文件列表本身不能证明 DRG 与 DFG 的真实配对。

```bash
python scripts/export_awa2_plot_features.py --sampling one-step
```

每类生成 100 个样本，同时导出两种 NPZ；种子为 9182，生成 batch size 为 100。
首组创建协议，后两组复用它。`one-step` 是这里示例选择的采样规则，应与 Rediff 一致；
如果统一使用逐步采样，改为 `--sampling progressive`。只导出一组时可添加
`--split-percent 100`（或 `30`、`10`）。

已有 Rediff 协议时，向三组传入同一份协议：

```bash
python scripts/export_awa2_plot_features.py --sampling one-step --protocol /absolute/path/to/Rediff/protocol.json
```

可添加 `--dry-run` 只打印命令，`--device cpu` 使用 CPU，或通过 `--output-root` 指定
另一输出根目录。正式运行会先检查所有选中权重和输出目录，避免缺少后续权重时启动导出；
输出目录已存在则报错。脚本不推断采样方式，不会按文件名自动切换到其他权重。

### 单次导出的完整参数

激活实际训练环境，在当前 ZeroDiff 项目根目录运行。若尚无共享协议，先导出一组，
建立第一份比较协议：

```bash
python -m feature_export.export \
  --dataset AWA2 --dataroot Dataset \
  --dfg-checkpoint PATH_TO_ZERODIFF_DFG.tar \
  --drg-checkpoint PATH_TO_MATCHING_DRG.tar \
  --method ZeroDiff --split-percent 100 \
  --view both --samples-per-class 100 \
  --sampling one-step --seed 9182 --batch-size 100 \
  --device cuda:0 --output-dir out/plot_features/AWA2/100pct
```

如需使用真实训练配置，加 `--training-config PATH_TO_CONFIG.json`。
CPU 导出使用 `--device cpu`；`auto` 会优先选 CUDA。

等价的 PowerShell 单行示例（填写实际路径）：

```powershell
python -m feature_export.export --dataset AWA2 --dataroot Dataset --dfg-checkpoint "PATH_TO_ZERODIFF_DFG.tar" --drg-checkpoint "PATH_TO_MATCHING_DRG.tar" --training-config "docs/feature_export_awa2.example.json" --method ZeroDiff --split-percent 100 --view both --samples-per-class 100 --sampling one-step --seed 9182 --batch-size 100 --device cuda:0 --output-dir out/plot_features/AWA2/100pct
```

如果你已在 Rediff 导出第一组，应改用下面的命令，复用那一份 `protocol.json` 的实际
绝对路径。两条命令是可选的起点，不要向同一个已存在的输出目录重复导出：

```bash
python -m feature_export.export \
  --dataset AWA2 --dataroot Dataset \
  --dfg-checkpoint PATH_TO_ZERODIFF_DFG.tar \
  --drg-checkpoint PATH_TO_MATCHING_DRG.tar \
  --method ZeroDiff --split-percent 100 \
  --view both --samples-per-class 100 \
  --sampling one-step --seed 9182 --batch-size 100 \
  --protocol /absolute/path/to/Rediff/100pct/seed9182/protocol.json \
  --device cuda:0 --output-dir out/plot_features/AWA2/100pct
```

Rediff 使用其项目中的导出器，设置 `--method Rediff`，加载它自己的 DFG 和实际配套
DRG，并复用同一份协议；两边谁先导出均可。本次修改不访问或修改 Rediff 项目。
本导出器仅实例化当前 `zerodiff_tools` 中的两个生成器与扩散系数，不导入会启动训练的
`zerodiff_DFG_train.py`。权重始终使用 `strict=True`；不兼容时会报告 DFG/DRG 角色、
文件及底层错误，需要核对训练配置和网络结构。

30% 和 10% 分别替换 DFG、DRG、训练配置、`--split-percent` 和输出目录，继续复用
最初的 `protocol.json`。方法名和训练比例不是协议锁定项，分别保存在 manifest；
数据文件、实际真实特征、类别名称及顺序、语义属性、预处理、生成数量、种子、
采样模式和 batch size 必须对齐，否则程序会在生成前报出具体不一致字段。

语义属性始终使用 `original_att`；是否进行 L2 归一化由训练配置的
`class_embedding_norm` 决定，与视觉 scaler 无关。

**协议哈希严格比较数据文件字节。** 即使 MAT 文件内容相似但被重新打包，也会拒绝
复用；建议两个项目使用相同的数据文件副本。如果不同训练比例用了不同特征提取器，
产生了不同的 `ce_ce.mat`，当前严格协议也会拒绝：需要先设计统一表征空间的实验，
不能仅凭维度都为 2048 直接叠加比较。

输出目录必须不存在，避免覆盖论文实验。程序完成全部文件写入后才发布该目录。
重复导出时，通过 `--output-dir` 指定新目录，例如
`out/plot_features/AWA2/100pct_repeat/`；不会自动覆盖原来的 `100pct/`。

## 4. 输出目录和 NPZ 字段

```text
out/plot_features/AWA2/100pct/
  generated_only.npz
  real_vs_generated.npz
  protocol.json
  manifest.json
```

两种 NPZ 字段完全一致；支持 `np.load(path, allow_pickle=False)`。

| 字段 | dtype / 形状 | 约定 |
| --- | --- | --- |
| `features` | float32 `[N, 2048]` | 原始高维 V，未执行 PCA/t-SNE |
| `labels` | int64 `[N]` | 原 AWA2 全数据集的零起始编号，即 MATLAB 编号减 1；不是重新编号的 0..9 |
| `source` | uint8 `[N]` | `0` 真实，`1` 生成 |
| `sample_ids` | Unicode `[N]` | `real:<dataset_index>` 或 `generated:<class_id>:<ordinal>` |
| `dataset_indices` | int64 `[N]` | 真实样本在原标签/特征表中的零起始行号；生成样本为 `-1` |
| `class_ids` | int64 `[10]` | 升序未见类编号；固定颜色时按此顺序映射 |
| `class_names` | Unicode `[10]` | 与 class_ids 一一对应 |
| `schema_version` | Unicode 标量 | `awa2-visual-features-v1` |

`generated_only`：按 class_ids 顺序，各类样本 ordinal 从 0 开始连续排列。
`real_vs_generated`：先是原 `test_unseen_loc` 顺序的全部真实样本，再是同一批生成样本。
请通过 `source` 划分来源，不要在绘图脚本里写死真实样本数量。

生成样本的 ID 是可配对的抽样位置，不表示不同方法生成了同一个特征；跨方法去重时
不能把同名生成 ID 合并，应使用 `(method, split_percent, sample_id)` 区分。
真实样本可以根据 `sample_ids` 和实际特征校验后去重。

## 5. JSON 元信息和复现边界

`protocol.json` 保存共享数据空间及采样规则，不含本地绝对路径、方法名或训练比例。
它同时校验原数据文件哈希和实际处理后特征、标签、索引、语义属性的数组哈希。

`manifest.json` 保存：接口版本、方法名、训练比例、创建时间（UTC）、特征空间、
DFG/DRG 路径与 SHA256、训练配置路径与 SHA256、有效推理参数、采样参数、
checkpoint 方法元信息及是否存在关系状态、数据文件来源、Git commit/工作区状态、
导出器及 `zerodiff_tools.py` 源码哈希、配置中使用默认值的字段、运行设备与依赖版本，
以及各 NPZ 的行数、文件和数组哈希。新增 manifest 字段不改变共享 protocol 或 NPZ schema。
`protocol_sha256` 是排序紧凑 JSON 的 SHA256，可用于核对整份协议。

每个类别独立使用 `(seed + 1000003 * class_id) % (2**63 - 1)` 作为种子。
四个初始噪声张量按原代码在 CPU 上依次抽样，逐步采样的后验噪声在推理设备上抽样；
CPU 情况共用该类的一个随机数流，CUDA 情况使用同种子但独立的设备随机数流。
这使配对方法拥有相同的噪声规则，并与模型初始化消耗的随机数分离。
批大小改变会改变随机数与样本的对应关系，所以也是协议锁定项。

同一设备、依赖版本、配置和权重下支持重复导出并得到相同数组；不保证 CPU/CUDA 或
不同库版本间逐位一致。时间戳、路径和 Git 状态可能变化，NPZ 文件字节也不作为
重复运行一致性的唯一依据；应比较数组或数组哈希。

## 6. 可视化项目如何读取和选择绘图

不需要导入 PyTorch 或两个训练项目，只需 NumPy 读取，再使用绘图库：

```python
import json
from pathlib import Path
import numpy as np

root = Path("inputs/AWA2/ZeroDiff/100pct/seed9182")
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
with np.load(root / "real_vs_generated.npz", allow_pickle=False) as z:
    assert z["schema_version"].item() == "awa2-visual-features-v1"
    X, y = z["features"], z["labels"]
    is_real = z["source"] == 0
    class_ids, class_names = z["class_ids"], z["class_names"]
```

- 第一张：读取每个方法的 `generated_only.npz`，固定类颜色，比较生成特征的类别结构。
- 第二张：读取 `real_vs_generated.npz`，同一面板把真实与生成特征合并运行一次
  PCA/t-SNE，然后依据 source 使用空心圆/星号。真实与生成不能各自降维后直接叠加。
- 可选择 PCA、UMAP、单类分布对比等；源特征不受最终图形选择约束。

绘图项目应先核对 schema 和 protocol 完全一致，再检查文件/数组哈希、shape、
每类数量、来源标记以及真实样本匹配。可以使用相同配置分别嵌入各面板，但独立嵌入的
坐标没有跨面板的共同含义，固定 random_state 也不会建立共同坐标系。
若要直接比较不同方法在同一坐标中的位置，应将所有组联合嵌入；真实样本只放入一次，
然后按组分面显示。联合嵌入与独立嵌入应在图注说明，并为所有组使用同一套参数。
第一张和第二张纳入的样本不同，所以即便复用生成特征，二维位置也可能不同。

100%/30%/10% 标题来自 manifest 的 split_percent。t-SNE 图作为定性证据，配合
分类结果分析；不要根据图形是否漂亮为不同方法选择不同随机种子或参数。

## 7. 验证

在已安装训练依赖及 pytest 的环境中，于本项目根目录运行：

```bash
python -m pytest tests/test_feature_export.py -q
python -m feature_export.export --help
```

Windows 本地若使用仓库内 `.venv` 测试环境，可将 `python` 替换为
`.\.venv\Scripts\python.exe`；服务器继续使用实际 ZeroDiff 训练环境。

`tests/test_feature_export.py` 用小型 AWA2 结构数据和随机权重检查三种训练比例的真实特征是否与原 DATA_LOADER
逐元素一致、CPU 单步/逐步采样是否与原训练脚本实际方法逐元素一致、两类 checkpoint、
三种导出模式、NPZ 无 pickle 读取、重复导出、协议失配拒绝和训练配置/DRG 配对校验。
采样测试仅通过 AST 提取原 `sample_from_model` 和 `sample_posterior` 方法来执行，
不会导入训练入口或启动训练；它不是另一份手写的参考采样实现。

这些是合成数据上的接口和数值一致性检查，不代表已经对真实 AWA2 模型验证过论文结果。
正式验收需要在训练服务器上使用实际数据、对应训练比例的 DFG 和配套 DRG：先导出一组，
再用相同参数和另一输出目录重复导出，检查 manifest 中的数组哈希一致，然后复用共享
protocol 导出其余方法和比例。
