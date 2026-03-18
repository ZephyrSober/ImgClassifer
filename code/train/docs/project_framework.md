# MindSpore 猫狗分类项目框架文档

## 1. 文档目标

本文档用于规划一个基于 MindSpore 的图像分类训练项目框架，面向当前的猫狗二分类任务，同时保留后续扩展到更多模型与更多数据集的空间。

当前阶段的目标不是直接实现训练代码，而是先把以下内容设计清楚：

- 项目目录结构
- 各模块职责边界
- 训练流程主线
- 配置组织方式
- 输入输出接口
- 实验管理与可维护性要求

第一期默认模型为 `ShuffleNetV2`，但整体框架不应与单一模型强绑定。

## 2. 当前项目上下文

根据仓库现状，训练框架应建立在现有数据工程产物之上，而不是重新设计数据预处理流程。

当前已具备：

- 数据清洗脚本：`code/src/data_clean.py`
- 数据划分脚本：`code/src/data_split.py`
- 划分结果 manifest：`dataset/splits/.../manifest.csv`
- 划分统计 summary：`dataset/splits/.../summary.json`

因此训练侧的设计原则是：

1. 训练代码统一消费 `manifest.csv`
2. 不再通过扫描原始目录推断 train/val/test
3. 不把数据清洗逻辑和训练逻辑混在一起

## 3. 总体设计原则

### 3.1 Manifest-first

训练阶段以 `manifest.csv` 为唯一数据源。  
训练代码只依赖结构化样本信息，不依赖目录扫描或日志解析。

### 3.2 配置驱动

模型、数据路径、训练参数、运行环境等都应由配置文件控制，而不是写死在 Python 脚本中。

### 3.3 模块解耦

数据读取、transform、模型构建、训练循环、评估、checkpoint 管理应独立分层，便于后续替换和扩展。

### 3.4 MindSpore 原生风格

框架设计应符合 MindSpore 的使用习惯：

- 模型使用 `nn.Cell`
- 数据通过 MindSpore Dataset 体系接入
- 训练步骤围绕 `forward`、`loss`、`grad`、`optimizer step` 组织
- 训练与评估逻辑分离

### 3.5 先保证闭环，再谈扩展

第一版框架应优先保证最小闭环清晰可实现：

- 加载数据
- 构建模型
- 训练
- 验证
- 测试
- 保存模型
- 保存实验结果

## 4. 建议目录结构

建议新增训练侧目录，但仅作为后续实现的规划蓝图：

```text
code/
├── data_src/
│   ├── data_clean.py
│   ├── data_split.py
│   └── log/
└── train/
    ├── docs/
    │   └── project_framework.md
    ├── configs/
    │   ├── dataset/
    │   ├── model/
    │   ├── train/
    │   ├── runtime/
    │   └── experiment/
    ├── src/
    │   ├── data/
    │   ├── models/
    │   ├── losses/
    │   ├── metrics/
    │   ├── engine/
    │   └── utils/
    └── tools/
```

## 5. 各目录职责规划

### 5.1 `code/train/docs/`

存放训练框架文档、实验约定、使用说明。  
该目录只放文档，不放代码实现。

### 5.2 `code/train/configs/`

负责所有实验配置。  
配置应分层组织，但第一版只保留训练运行真正需要的最小字段，避免把还没实现的扩展能力提前写进配置。

第一版建议分层如下：

- `dataset/`
  - `dataset_root`
  - `manifest_path`
  - `label_map`
  - `image_size`
  - `normalize`
- `model/`
  - `model_name`
  - `num_classes`
  - `pretrained`
- `train/`
  - `epochs`
  - `batch_size`
  - `optimizer`
  - `loss`
  - `log_interval`
- `runtime/`
  - `device_target`
  - `device_id`
  - `seed`
  - `num_parallel_workers`
  - `run_root`
- `experiment/`
  - `experiment_name`
  - `dataset_config`
  - `model_config`
  - `train_config`
  - `runtime_config`

配置层还应遵循以下最简原则：

- 只放训练运行必须提供、且可能变化的实验参数
- 不把 manifest 中已经存在的数据事实重复写进配置
- 不为当前还没有实现的扩展能力提前增加字段
- 第一版默认指标固定为 `accuracy`
- 第一版默认保存 `best.ckpt` 和 `last.ckpt`

### 5.3 `code/train/src/data/`

负责训练数据接入，建议仅承担以下职责：

- 读取 manifest
- 过滤 train/val/test
- 构造标签映射
- 解析图片路径
- 定义 transform
- 构建 MindSpore Dataset
- 完成 shuffle、batch、repeat

不建议在该层做：

- 模型相关逻辑
- loss 相关逻辑
- 训练日志聚合

### 5.4 `code/train/src/models/`

负责模型构建与注册。

建议职责：

- 提供统一模型构建入口
- 通过模型名创建网络
- 支持后续扩展更多 backbone

第一版默认支持：

- `shufflenet_v2_x1_0`

不建议把训练参数、损失函数、优化器写进模型模块。

### 5.5 `code/train/src/losses/`

负责损失函数构建。  
第一版仅需要交叉熵损失即可。

该层存在的意义是后续若增加：

- label smoothing
- focal loss
- class-balanced loss

无需改动训练主循环。

### 5.6 `code/train/src/metrics/`

负责训练和评估阶段的指标计算。

第一版最少应包含：

- accuracy

后续可扩展：

- precision
- recall
- F1
- confusion matrix

### 5.7 `code/train/src/engine/`

负责训练引擎与评估引擎，是训练框架的核心流程层。

建议拆分为：

- `train_one_epoch`
- `validate_one_epoch`
- `test_one_epoch`
- checkpoint 保存逻辑
- best model 选择逻辑

该层负责组织流程，但不负责定义模型结构或硬编码数据路径。

### 5.8 `code/train/src/utils/`

负责与训练主逻辑无关但会重复使用的通用能力：

- 配置读取
- 路径解析
- 日志初始化
- 随机种子设置
- checkpoint 路径管理
- run 目录创建
- 配置快照保存

### 5.9 `code/train/tools/`

负责命令行入口。

建议只放薄脚本，例如：

- `train.py`
- `eval.py`
- `export.py`

这些脚本只负责：

- 读取命令行参数
- 加载配置
- 调用核心模块

不应把完整训练逻辑直接写在脚本中。

## 6. 训练流程主线设计

整个训练过程建议严格遵循以下顺序：

### 6.1 加载配置

训练入口先读取 experiment 配置，再解析出：

- dataset 配置
- model 配置
- train 配置
- runtime 配置

### 6.2 解析数据路径

需要从配置中得到：

- 图片根目录
- manifest 路径
- 输出目录

路径解析应在启动阶段一次完成，避免训练中动态推断。

### 6.3 加载 manifest

训练代码从 manifest 中读取样本记录，并按 `split` 字段筛分：

- `train`
- `val`
- `test`

### 6.4 构造标签映射

对于当前猫狗分类任务，第一版约定：

- `Cat -> 0`
- `Dog -> 1`

文档层面建议把这个约定写清楚，避免训练和推理两端标签顺序不一致。

### 6.5 定义 transform

建议按数据阶段区分：

- `train`：
  - resize
  - random crop 或 center crop 中的训练版本
  - random horizontal flip
  - normalize
- `val/test`：
  - resize
  - center crop
  - normalize

第一版不引入过多增强，以保证 baseline 结果稳定、可解释。

### 6.6 构建 MindSpore Dataset

训练阶段推荐使用 MindSpore 的数据集接口包装 manifest 样本。

这里应完成：

- 样本生成
- transform 应用
- label 转换
- shuffle
- batch

是否 `repeat` 可由后续实现决定，但第一版不是必须复杂化。

### 6.7 构建模型

模型构建统一通过模型工厂完成。  
第一版默认模型为 `ShuffleNetV2`，输出类别数固定为 2。

### 6.8 构建 loss 和 optimizer

建议第一版采用：

- loss：交叉熵
- optimizer：SGD

选择理由：

- 作为图像分类 baseline 足够稳定
- 配置简单
- 便于后续和其他实验对比

### 6.9 前向传播与 loss 计算

每个 batch 的训练步骤应包含：

1. 读取图像和标签
2. 模型前向传播得到 logits
3. 计算 loss

这里建议将 `forward` 与 `loss` 组织为清晰的单元，而不是在训练循环里写成过长的内联逻辑。

### 6.10 梯度计算与参数更新

训练步骤继续包含：

4. 计算梯度
5. 调用 optimizer 更新参数

文档层面只需规定“训练引擎显式管理梯度与参数更新”，不必提前写死具体代码风格。

### 6.11 模型验证

每个 epoch 完成后都应执行验证流程。  
验证阶段要求：

- 不更新参数
- 只统计 loss 和 accuracy
- 作为 best model 判断依据

### 6.12 模型测试

测试集只在训练完成后使用一次。  
其职责是报告最终结果，而不是参与调参。

### 6.13 模型保存

建议至少保存两类 checkpoint：

- `last.ckpt`
  - 当前最新训练状态
- `best.ckpt`
  - 验证集 accuracy 最优模型

### 6.14 结果落盘

每次实验建议输出：

- config 快照
- 训练日志
- 每 epoch 指标
- 最佳 epoch
- best checkpoint 路径
- final test metrics

## 7. 对外接口规划

为了让实现代码在后续保持稳定，建议提前约定以下接口。

### 7.1 数据层接口

- `build_label_map(labels) -> dict[str, int]`
- `load_manifest(manifest_path, split=None) -> list[SampleRecord]`
- `build_transforms(stage, image_size, normalize_cfg)`
- `build_dataset(records, dataset_root, stage, config)`
- `build_dataloader(split, config)`

### 7.2 模型层接口

- `build_model(config) -> nn.Cell`

其中 `config` 至少包含：

- 模型名
- 类别数
- 是否使用预训练

### 7.3 训练引擎接口

- `train_one_epoch(...)`
- `validate_one_epoch(...)`
- `test_one_epoch(...)`
- `save_checkpoint_if_best(...)`

### 7.4 CLI 入口接口

- `train.py --config <experiment.yaml>`
- `eval.py --config <experiment.yaml> --ckpt <path>`
- `export.py --config <experiment.yaml> --ckpt <path>`

## 8. 样本结构设计

训练阶段建议定义统一的样本记录结构 `SampleRecord`。

最小必要字段：

- `relative_path`
- `label`
- `split`

推荐保留的扩展字段：

- `width`
- `height`
- `size_bucket`
- `aspect_bucket`
- `difficulty_tag`
- `was_converted_to_rgb`

这样做的好处是：

- baseline 可以只用最小字段
- 后续做难例分析时无需改 manifest 读取接口
- 后续做分层评估时可以直接复用数据字段

## 9. 配置组织建议

推荐使用多层配置，而不是一个大配置文件包含所有内容。  
但第一版配置应坚持“必须项优先”，以便阅读和实现都尽量直接。

### 9.0 如何理解配置

为了避免把不同类型的信息混在一起，建议先区分三类内容：

- 数据事实
  - 例如 `width`、`height`、`split`、`label`
  - 这些信息来自 `manifest.csv`，不应重复写进训练配置
- 实验参数
  - 例如 `image_size`、`epochs`、`lr`
  - 这些信息会影响一次训练运行，应写进配置
- 框架默认行为
  - 例如第一版指标固定为 `accuracy`
  - 如果当前不需要切换，就先不暴露为配置项

第一版建议结构如下：

### 9.1 dataset 配置

负责：

- `dataset_root`
- `manifest_path`
- `label_map`
- `image_size`
- `normalize`

其中 `image_size` 表示训练输入尺寸，不表示原始图片尺寸。  
原始图片的真实宽高应来自 `manifest.csv` 中的 `width` 和 `height` 字段。

### 9.2 model 配置

负责：

- `model_name`
- `num_classes`
- `pretrained`

### 9.3 train 配置

负责：

- `epochs`
- `batch_size`
- `optimizer`
- `loss`
- `log_interval`

### 9.4 runtime 配置

负责：

- `device_target`
- `device_id`
- `seed`
- `num_parallel_workers`
- `run_root`

### 9.5 experiment 配置

负责组合上述配置，形成一次完整实验的入口：

- `experiment_name`
- `dataset_config`
- `model_config`
- `train_config`
- `runtime_config`

## 10. 实验目录规划

建议每次运行生成独立实验目录：

```text
runs/<timestamp>_<experiment_name>/
├── checkpoints/
├── logs/
├── metrics/
└── config_snapshot.yaml
```

各目录建议职责：

- `checkpoints/`
  - `best.ckpt`
  - `last.ckpt`
- `logs/`
  - 训练日志
  - 验证日志
- `metrics/`
  - 历史指标
  - 最终测试结果
- `config_snapshot.yaml`
  - 当次实验完整配置快照

## 11. 可维护性要求

### 11.1 不把逻辑写死在脚本里

CLI 入口只负责调用，不负责承载核心业务逻辑。

### 11.2 不让模型耦合数据层

模型只关心输入张量和输出类别数，不关心 manifest 路径、标签字符串等外部细节。

### 11.3 不让数据层承担训练逻辑

数据层负责样本准备，不负责 loss、optimizer、metric 统计。

### 11.4 配置优先于硬编码

只要属于“实验参数”，都应优先进入配置层。

### 11.5 面向扩展而不是面向当前一次实验写死

即使当前只有猫狗二分类，也要保留未来扩展空间：

- 多分类
- 新模型
- 新数据集
- 新评估指标

## 12. 第一版基线建议

为了让项目尽快具备可实施性，第一版架构文档默认如下约定：

- 模型：`ShuffleNetV2 x1.0`
- 输入尺寸：`224`
- 类别数：`2`
- 优化器：`SGD`
- 损失函数：交叉熵
- 指标：accuracy
- 数据输入：当前 manifest 划分结果

该约定的意义是：

- 降低实现复杂度
- 快速搭建第一条完整训练链路
- 为后续换更强模型提供稳定对照 baseline

## 13. 文档结论

这套训练框架的核心思想是：

- 用现有 manifest 结果驱动训练
- 用 MindSpore 风格组织训练流程
- 用清晰分层保证后续可维护性
- 用配置管理实验，而不是把实验逻辑写死在代码里

后续如果进入实现阶段，应严格以本文档为蓝图，先搭最小闭环，再逐步扩展模型、增强策略和实验功能。
