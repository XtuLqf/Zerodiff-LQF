# ZeroDiff 最简复现实验指南

本文档按当前实验设置固定使用 `att` 语义嵌入，不再使用 `sent`。

## 0. 服务器信息与环境准备

### 0.1 服务器信息

- GPU: RTX 5090
- Driver: 580.126.09
- System CUDA: 13.0

说明：推荐对齐到已验证可用组合：PyTorch 2.9.1+cu130。

### 0.2 创建虚拟环境（名称固定：zerodiff）

```bash
conda create -n zerodiff python=3.10 -y
conda activate zerodiff
python -m pip install --upgrade pip
```

### 0.3 安装 PyTorch（cu130）

```bash
pip install torch==2.9.1+cu130 torchvision==0.24.1+cu130 torchaudio==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
```

### 0.4 安装实验依赖

```bash
pip install scikit-learn==1.3.0 scipy==1.10.0 numpy==1.24.3 pillow==9.4.0
```

### 0.5 验证环境

```bash
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0))"
```

### 0.6 单卡运行约定

本机单卡训练统一使用：

```bash
export CUDA_VISIBLE_DEVICES=0
```

## 1. 按 README.md 理解训练流程

`README.md` 的训练说明是：

1. 先训练 `DRG`
2. 再训练 `DFG`
3. 训练和评估入口以 `scripts/run_*_zerodiff_{DRG,DFG}_train.py` 为准

也就是说，`README.md` 并不是让你自己手写一条简化命令去猜参数，而是让你直接运行仓库里已经给好的脚本。

当前仓库中可用的官方入口如下：

- `scripts/run_cub_zerodiff_DRG_train.py`
- `scripts/run_cub_zerodiff_DFG_train.py`
- `scripts/run_awa2_zerodiff_DRG_train.py`
- `scripts/run_awa2_zerodiff_DFG_train.py`
- `scripts/run_sun_zerodiff_DRG_train.py`
- `scripts/run_sun_zerodiff_DFG_train.py`

## 2. 数据准备与检查

数据组织方式需要满足 `README.md` 和 `datasets/image_util.py` 的读取要求。

当前仓库已经按项目内相对路径创建了这些目录：

```text
Dataset/
├── AWA2/
├── CUB/
├── SUN/
log/
├── AWA2/
├── CUB/
├── SUN/
out/
├── AWA2/
├── CUB/
├── SUN/
FineTune/PACO/checkpoints/
```

对每个数据集，至少要准备以下文件：

- `Dataset/<DATASET>/res101.mat`
- `Dataset/<DATASET>/att_splits.mat`
- `Dataset/<DATASET>/ce_ce.mat`
- `Dataset/<DATASET>/con_paco.mat`

如果要做低比例训练，还需要：

- `Dataset/<DATASET>/split_10percent.mat`
- `Dataset/<DATASET>/split_30percent.mat`

如果你要自己重新生成特征，而不是直接下载作者提供的 fine-tuned features，还需要补这几类文件：

- `Dataset/AWA2/Animals_with_Attributes2/...`
- `Dataset/CUB/CUB_200_2011/...`
- `Dataset/SUN/images/...`
- `FineTune/PACO/checkpoints/paco_r101.pth`
- `pretrained_models/resnet101-5d3b4d8f.pth`

对应关系如下：

- `con_paco.mat` 最终放到 `Dataset/<DATASET>/con_paco.mat`
- `ce_ce.mat` 最终放到 `Dataset/<DATASET>/ce_ce.mat`
- PACO checkpoint 放到 `FineTune/PACO/checkpoints/paco_r101.pth`
- ResNet101 预训练权重放到 `pretrained_models/resnet101-5d3b4d8f.pth`

可以在项目根目录执行检查：

```bash
bash scripts/check_dataset_mats.sh --dataroot ./Dataset --class-embedding att
```

如果还要检查低比例实验所需文件：

```bash
bash scripts/check_dataset_mats.sh --dataroot ./Dataset --class-embedding att --check-split
```

## 3. 推荐运行步骤

必须在项目根目录运行，避免相对路径偏移：

```bash
cd d:/GitCode/Zerodiff-LQF
```

标准流程固定为两步：

### 3.1 第一步：训练 DRG

按数据集选择对应脚本：

```bash
python ./scripts/run_cub_zerodiff_DRG_train.py
```

或：

```bash
python ./scripts/run_awa2_zerodiff_DRG_train.py
```

或：

```bash
python ./scripts/run_sun_zerodiff_DRG_train.py
```

这一阶段会训练第一阶段模型，并在 `out/<dataset>/` 下生成 DRG checkpoint。

### 3.2 第二步：训练 DFG

在 DRG 训练完成后，再运行同数据集对应的 DFG 脚本：

```bash
python ./scripts/run_cub_zerodiff_DFG_train.py
```

或：

```bash
python ./scripts/run_awa2_zerodiff_DFG_train.py
```

或：

```bash
python ./scripts/run_sun_zerodiff_DFG_train.py
```

DFG 脚本内部会通过 `--netR_model_path` 指向前一步 DRG 产生的权重文件，所以：

- DRG 和 DFG 必须使用同一数据集
- DRG 和 DFG 必须使用同一组相容超参数
- 如果你改了脚本里的参数，也要同步修改 DFG 的 `--netR_model_path`

## 4. 训练过程中会自动完成什么

按 `README.md` 的流程运行后，不需要再单独写额外评估命令。

训练脚本会自动完成：

- 模型训练
- 日志记录
- ZSL 评估
- GZSL 评估
- `latest` / `best_zsl` / `best_gzsl` 权重保存

因此，复现实验的最小闭环就是：

1. 准备环境
2. 准备数据
3. 运行 DRG 脚本
4. 运行 DFG 脚本
5. 在日志和输出目录中查看结果

## 5. 输出位置

- 日志目录：`./log/<dataset>/`
- 权重目录：`./out/<dataset>/`

常见输出包括：

- DRG 权重
- DFG 的 `latest` 权重
- DFG 的 `best_zsl` 权重
- DFG 的 `best_gzsl` 权重

## 6. 注意事项

- 本文档以 `README.md` 的流程为准，不再单独发明一套与 `scripts/run_*` 不一致的简化命令。
- 如果你手动直接运行 `zerodiff_DRG_train.py` 或 `zerodiff_DFG_train.py`，需要自己完整对齐对应 `scripts/run_*` 中的参数；否则结果可能和作者仓库默认复现结果差异很大。
- `config_zerodiff.py` 里的默认值是参数默认值，不等于各数据集的最终复现配置。
- 当前仓库已经把 six 个 `scripts/run_*` 入口统一到项目内的 `./Dataset` 路径，并且 CUB 也统一切换到了 `att` 方案。
- 当前仓库已经修复 `SUN DRG` 脚本误调用 `zerodiff_DFG_train.py` 的问题。
- 当前仓库已经兼容现代 PyTorch：`DRG` 和 `DFG` 入口中的旧版 `volatile` 推理写法已替换，`DRG` 评估阶段的张量占位复制也不再使用不兼容的 `Tensor.copy()`；这属于代码兼容性修复，不需要修改 `zerodiff` 虚拟环境。
- `zerodiff_DRG_train.py` 保存的 checkpoint 现在同时兼容 `state_dict_R` 和 `state_dict_G_con`，`zerodiff_DFG_train.py` 也会自动兼容这两种键名。
