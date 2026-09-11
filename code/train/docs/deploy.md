# 模型部署到手机端（导出 `.ms`）

本仓库从「训练好的 ShuffleNetV2」到「手机 App 可加载的 `.ms`」的完整链路：

```
训练/已有 ckpt  →  export.py 导出  →  .mindir  →  converter_lite 转换  →  .ms
```

## 0. 前置条件

- 训练环境装好 MindSpore（本仓库在 **MindSpore 1.9.0 / Python 3.7** 上验证通过）。
- 有一份 ShuffleNetV2 的 `.ckpt`（用本仓库自己训练，见 [train.py](../tools/train.py) + `cats_dogs_shufflenet_v2_cpu.yaml`）。
- 下载 **MindSpore Lite 1.9.0** 的 Windows 包 `mindspore-lite-1.9.0-win-x64.zip`，把其中的
  `tools/converter/` 整个目录复制到仓库根目录、重命名为 `converter/`（这样
  `converter/converter/converter_lite.exe` + `converter/lib/*.dll` 就位）。
  注意：`converter/` 已在 `.gitignore` 中、**不随 git 提交**，所以每个同学都要自己下载解压一次。
  （`converter_lite` 版本必须和训练用的 MindSpore 版本对应，即 1.9.x。）

## 1. 导出 `.mindir`

```bash
python code/train/tools/export.py --ckpt <你的.ckpt路径>
```

默认导出 ShuffleNetV2（`--model shufflenet_v2_x1_0`），输出 `shufflenetv2.mindir`。要导出别的模型时指定：

```bash
python code/train/tools/export.py --ckpt <你的.ckpt路径> --model simple_cnn --output simple_cnn
```

关键点：

- 导出图固定输入 `[1, 3, 224, 224]`（float32、NCHW）。
- 导出图包含 **Softmax**，输出 `[1, 2]`（index 0 = Cat，index 1 = Dog），
  和手机 App 的输入/输出契约一致。
- 导出在 `GRAPH_MODE` 下进行（`export.py` 内部已设置）。

## 2. 转换 `.ms`

```bash
python code/train/tools/convert.py
```

`--converter`、`--model`、`--output` 都有默认值（分别是 `converter/converter/converter_lite.exe`、
`shufflenetv2.mindir`、`pet`）。注意 `converter/` **不随 git 提交**，需按「前置条件」自行下载解压到
仓库根目录，否则 convert.py 会报找不到 converter。转别的模型时手动指定即可：

```bash
python code/train/tools/convert.py --model <模型.mindir> --output <输出名>
```

等价于直接运行：

```bash
converter/converter/converter_lite.exe --fmk=MINDIR --modelFile=shufflenetv2.mindir --outputFile=pet
```

成功后生成 `pet.ms`，看到 `CONVERT RESULT SUCCESS:0` 即成功。

> 两个常见坑：
> 1. **converter_lite 版本必须和训练用的 MindSpore 版本配套**（1.9 配 1.9）。原项目自带的那个 `converter_lite.exe`（2021 年，约 57MB）是老的 1.1.1 版，**转不了 1.9 导出的 MindIR**，会报 `Build funcgraph failed`。
> 2. 1.9.0 的 `converter_lite.exe` 依赖 `tools/converter/lib/` 下的 DLL，直接跑会报 `libmindspore_converter.dll 找不到`。用 `convert.py` 已自动把 lib 目录加进 PATH，无需手动处理；如果直接跑 converter_lite，需先把该 lib 目录加入 PATH。

## 3. 放入手机 App

把 `pet.ms` 放进手机 App（原项目里叫 `PetClassification`）的 assets 目录，App 端
用 MindSpore Lite 加载即可做猫狗识别。

## 输入/输出契约（务必与手机端对齐）

| 项 | 约定 |
|---|---|
| 输入 | `[1, 3, 224, 224]` float32，NCHW，归一化后的 RGB（ImageNet mean/std：`0.485/0.456/0.406`、`0.229/0.224/0.225`） |
| 归一化 | 在手机端做：`(像素/255 - mean) / std`，再 HWC→CHW |
| 输出 | `[1, 2]` softmax 概率，`argmax` 后 0=Cat、1=Dog |

归一化这一环在模型图之外（训练时的 transform 也在模型外），手机端必须做同样的
resize + 归一化，否则精度会明显下降。
