# SAM3_AGXOrin

在 **NVIDIA Jetson AGX Orin (64GB)** 上部署 **Meta SAM3**（Segment Anything Model 3），
面向**自然语言驱动的图像分割**：直接用文字描述目标（`"dog"`、`"the red car"`、`"shoes"`）
即可得到像素级掩码，无需人工画框或点选；同时保留点/框提示与 TensorRT 加速能力。

本项目在官方 `sam3` 包之上做了三类工程化工作：

1. **Jetson 适配** —— 解决 aarch64 上无 Triton 导致的 `import sam3` 失败；
2. **三种提示 + 结果后处理** —— 文本 / 框 / 点提示统一命令行，点提示自动「点谁分谁」；
3. **常驻服务与 TensorRT 加速** —— 模型只加载一次，单次推理从 **19s 降到 1.2s**（约 15×）。

---

## 目录

1. [硬件与环境](#1-硬件与环境)
2. [快速开始](#2-快速开始)
3. [三种提示方式](#3-三种提示方式)
4. [常驻服务（推荐）](#4-常驻服务推荐)
5. [TensorRT 加速](#5-tensorrt-加速)
6. [参数速查](#6-参数速查)
7. [项目结构](#7-项目结构)
8. [实现要点](#8-实现要点)
9. [Jetson 关键适配：Triton 问题](#9-jetson-关键适配triton-问题)
10. [性能实测](#10-性能实测)
11. [资源占用与健康](#11-资源占用与健康)
12. [常见问题](#12-常见问题)
13. [许可](#13-许可)

---

## 1. 硬件与环境

| 项目 | 值 |
|------|-----|
| 设备 | Jetson AGX Orin 64GB（统一内存架构，CPU/GPU 共享） |
| 系统 | Linux 5.15（JetPack 6.x / Ubuntu 22.04） |
| Python | 3.10.12（`venv310/`） |
| PyTorch | 2.8.0（NVIDIA 定制 aarch64 wheel） |
| torchvision | 0.23.0 |
| CUDA | 12.6 可用 |
| TensorRT | 10.3（用于可选的 engine 加速） |
| sam3 | 0.1.4（PyPI，`--no-deps` 安装） |
| 权重 | `models/sam3.pt`（3.3 GB） |

从零初始化（新机器）：

```bash
cd /home/jetson/zhangtianqi/SAM3_AGXOrin
./sh/setup.sh       # 建 venv + 装依赖 + 打补丁 + 下载权重 + 自检
./sh/download.sh    # 仅重新下载权重/词表（断点续传）
./sh/check_env.sh   # 检查 ONNX / TensorRT / onnxruntime 可用性
```

环境自检：

```bash
./venv310/bin/python py/selfcheck.py
```

---

## 2. 快速开始

> 所有命令都在项目根目录执行：`cd /home/jetson/zhangtianqi/SAM3_AGXOrin`

```bash
# 文本提示分割（核心功能）
./sh/run.sh --image inputimage/6.jpg --text "shoe" --out outputimage/6_text.png

# 点提示分割（点谁分谁）
./sh/run.sh --image inputimage/5.jpg --point 114 285 --out outputimage/5_point.png

# 框提示分割（兼容 SAM1 风格）
./sh/run.sh --image inputimage/6.jpg --box 100 250 300 200 --out outputimage/6_box.png

# 只保留掩码，不要目标框
./sh/run.sh --image inputimage/2.jpg --point 417 344 --no-box --out outputimage/2_mask.png

# 导出每个掩码为独立 PNG
./sh/run.sh --image inputimage/6.jpg --text "shoe" \
            --out outputimage/6_text.png --mask-dir outputs/006_masks

# 批量处理（模型只加载一次）
./venv310/bin/python py/infer_batch.py --input inputimage --text "person" \
            --outdir outputimage/batch_person --csv outputs/person.csv

# 性能基准
./sh/run.sh --image inputimage/6.jpg --text "shoe" --bench --repeat 5
```

---

## 3. 三种提示方式

三种提示可任选其一（同时给出时优先级为 `--text` > `--point` > `--box`）。

### 3.1 文本提示 `--text`

输入自然语言描述，支持开放词汇与短语：

```bash
./sh/run.sh --image inputimage/6.jpg --text "shoe"      --out out.png
./sh/run.sh --image inputimage/1.jpg --text "the dog"   --out out.png
./sh/run.sh --image inputimage/4.jpg --text "red car"   --out out.png
```

### 3.2 点提示 `--point`

坐标是**像素坐标**（脚本内部自动归一化到 `[0,1]`，这是 SAM3 几何编码器的要求）。
可给多个点，并用 `--point-label` 指定正/负样本（`1` 正 / `0` 负，默认全 `1`）：

```bash
# 单点
./sh/run.sh --image inputimage/5.jpg --point 114 285 --out out.png

# 多点：同时对两个物体分割
./sh/run.sh --image inputimage/6.jpg --point 490 167 384 90 --out out.png

# 正 + 负点：保留正点目标，排除负点所在区域
./sh/run.sh --image inputimage/6.jpg --point 490 167 100 400 \
            --point-label 1 0 --out out.png
```

**点提示的「点谁分谁」后处理**（本项目增强，默认开启）：

SAM3 的 grounding 头会一次性输出画面中所有候选物体（点只是「几何提示」之一），
因此直接跑点提示会得到很多框。本项目推理后自动做两步筛选：

1. **按点筛选** —— 只保留掩码中**包含正样本点**的目标（默认行为）；
2. **去嵌套** —— 若某个大掩码几乎完整包住了已保留的小掩码（如「桌面」包住「杯垫」，
   重叠率 > 90%），丢弃大掩码，只留下点所在的**最具体**物体。

```bash
# 想要原始行为（全部候选）：
./sh/run.sh --image inputimage/2.jpg --point 417 344 --keep-all --out out.png

# 只画掩码、不画框和分数：
./sh/run.sh --image inputimage/2.jpg --point 417 344 --no-box --out out.png
```

实测（`inputimage/2.jpg`，点 `(417,344)`）：候选 **12 个 → 筛选后 1 个**
（正是点所在的杯垫，面积 3103px），另有一个包住它的桌面区域（15036px）被去嵌套丢弃。

### 3.3 框提示 `--box`

格式为 `X Y W H`（左上角坐标 + 宽高，像素），`--label` 指定正/负样本：

```bash
./sh/run.sh --image inputimage/6.jpg --box 100 250 300 200 --out out.png
```

### 3.4 可视化说明

输出图上会标注：

- **目标框** + `序号:置信度`（可用 `--no-box` 关闭）
- **掩码**叠加（颜色来自内置调色板，`--alpha` 调透明度）
- **提示点**：正样本为**绿十字**，负样本为**红十字**（仅点提示时绘制）

---

## 4. 常驻服务（推荐）

### 4.1 为什么需要

单次运行 `./sh/run.sh ...` 时，**大部分时间花在加载模型**（约 16s），真正的推理只有 1~2s：

```
[计时] 初始化: 导入三方库 1.7s | 模型加载 14.7s | 初始化总计 16.5s
[计时] 热身后推理: 图像编码 814ms | 提示前向 282ms | 合计 1096ms
```

常驻服务把模型留在内存里，后续每次请求只做「图像编码 + 提示前向」。

### 4.2 用法

```bash
# 启动服务（模型加载一次，约 18s）
./sh/serve.sh start
./sh/serve.sh start --warmup               # 启动时预热一张图，首帧更快
./sh/serve.sh start --idle-timeout 1800    # 空闲 30 分钟自动退出释放显存

# 之后正常推理 —— 命令完全不变，约 1.2s 返回
./sh/run.sh --image inputimage/5.jpg --point 114 285 --out outputimage/5_out_point.png
./sh/run.sh --image inputimage/6.jpg --text "shoe"   --out outputimage/6_out.png

# 管理
./sh/serve.sh status      # 查看服务状态 + 显存占用
./sh/serve.sh log         # 跟踪服务日志
./sh/serve.sh stop        # 停止并释放模型与显存
./sh/serve.sh restart     # 重启
./sh/serve.sh fore        # 前台启动（Ctrl+C 即释放）
```

### 4.3 工作原理

```
./sh/run.sh ...  ->  py/sam3_client.py
                        ├─ run/sam3.sock 存在且服务在线 -> 转发（约 1.2s）
                        └─ 否则                        -> 本地加载单次运行（约 16s）
```

- **通信**：Unix domain socket（默认 `run/sam3.sock`），一行 JSON 请求 / 一行 JSON 响应；
- **协议**：`{"cmd":"infer","argv":[...]}`、`{"cmd":"ping"}`、`{"cmd":"stop"}`；
- **对使用者透明**：`run.sh` 的参数与用法**完全不变**；
- **可选优化**：不启动服务也不会报错，客户端会自动回退到本地运行；
- **同一套代码**：服务端与本地运行都调用 `py/sam3_infer.py` 的
  `prepare_args()` / `run_image()` / `report()`，因此**输出必然一致**。

### 4.4 释放显存的三种方式

| 方式 | 命令 | 说明 |
|------|------|------|
| 主动释放 | `./sh/serve.sh stop` | 优雅通知服务退出，释放显存并清理 socket/pid |
| 自动释放 | `./sh/serve.sh start --idle-timeout 1800` | 空闲超时后自动退出 |
| 手动中断 | `./sh/serve.sh fore` 后按 `Ctrl+C` | 前台模式 |

环境变量：

| 变量 | 说明 |
|------|------|
| `SAM3_SOCKET` | 自定义 socket 路径（客户端与服务端须一致） |
| `SAM3_NO_SERVER` | 设为 `1` 强制本地运行（绕过服务） |

---

## 5. TensorRT 加速

将视觉编码器与文本编码器分别导出 ONNX 并构建 TensorRT engine，可绕过部分
PyTorch 运行时开销。

```bash
# 1) 导出 ONNX（生成 models/*.onnx）
./venv310/bin/python py/export_onnx_sam3.py

# 2) 构建 FP32 engine（生成 models/*.engine）
./sh/build_engine_sam3.sh models/sam3_vision_encoder.onnx
./sh/build_engine_sam3.sh models/sam3_text_encoder.onnx

# 3) 用 engine 推理（参数与 run.sh 完全一致）
./sh/run_trt.sh --image inputimage/6.jpg --text "shoe" --out outputimage/6_trt.png

# 4) 精度/一致性校验
./venv310/bin/python py/verify_onnx_cpu.py     # ONNX 与 PyTorch 对齐
./venv310/bin/python py/verify_trt_prec.py     # TRT 与 PyTorch 对齐

# A/B 对比：单独回退某一侧到 PyTorch
./sh/run_trt.sh --image a.jpg --text "dog" \
                --vision-backend pytorch --text-backend engine --out out.png
```

**实测结论：在 AGX Orin 上 TRT 对本模型没有明显加速。**

| 后端 | 首次推理 | 热身后推理 | 结果一致性 |
|------|---------|-----------|-----------|
| PyTorch | 1463 ms | 1065 ms | — |
| TensorRT | 2523 ms | 1062 ms | ✅ 完全一致（同 score/box） |

原因：模型瓶颈在 ViT-H 视觉编码与融合 Transformer 的**访存**，而 Orin 的
Tensor Core 对该形状矩阵乘的收益有限；此外 TRT 首次运行有引擎加载与
CUDA 图初始化开销。**日常使用建议直接用 PyTorch 路径（`run.sh`）**。

> TRT engine 与 ONNX 各约 3.1 GB，若磁盘紧张可删除 `models/*.onnx`，
> 保留 `*.engine` 即可运行 `run_trt.sh`。

---

## 6. 参数速查

### 6.1 推理参数（`sh/run.sh` 与 `sh/run_trt.sh` 通用）

| 参数 | 说明 | 默认 |
|------|------|------|
| `--image` | 输入图像路径（必需） | — |
| `--text` | 文本提示，如 `"dog"`、`"red car"` | — |
| `--box X Y W H` | 框提示：左上角坐标 + 宽高（像素） | — |
| `--label` | 框提示标签：`1` 正 / `0` 负 | `1` |
| `--point X Y [X Y ...]` | 点提示：像素坐标，成对给出，可多对 | — |
| `--point-label L [L ...]` | 每点标签：`1` 正 / `0` 负，数量须与点一致 | 全 `1` |
| `--no-box` | 只绘制掩码，不绘制目标框与分数 | 关 |
| `--keep-all` | 点提示时保留全部候选（关闭筛选与去嵌套） | 关 |
| `--out` | 可视化输出路径 | — |
| `--mask-dir` | 每个掩码单独导出 PNG 到此目录 | — |
| `--threshold` | 置信度阈值 | `0.5` |
| `--max-masks` | 最多可视化前 N 个 | `20` |
| `--alpha` | 掩码叠加透明度 | `0.5` |
| `--bench` / `--repeat` | 打印耗时基准 / 重复次数 | 关 / `5` |
| `--bpe` / `--ckpt` | 自定义 BPE 词表 / 权重路径 | `assets/`、`models/` |
| `--device` | `cuda` 或 `cpu` | `cuda` |
| `--no-tf32` | 关闭 TF32（默认开启加速） | 关 |

`run_trt.sh` 额外支持 `--vision-backend {engine,pytorch}` 与 `--text-backend {engine,pytorch}`。

> 查看参数帮助请用：`./venv310/bin/python py/sam3_infer.py --help`
> （`./sh/run.sh --help` 缺少 `--image` 会被参数校验拦截。）

### 6.2 服务参数（`sh/serve.sh`）

| 子命令 | 说明 |
|--------|------|
| `start [参数]` | 后台启动服务 |
| `stop` | 停止并释放显存 |
| `restart [参数]` | 重启 |
| `status` | 查看状态与显存占用 |
| `log` | 跟踪日志（`run/sam3.log`） |
| `fore [参数]` | 前台启动 |

可透传给服务的参数：`--idle-timeout N`、`--threshold F`、`--warmup`、
`--socket PATH`、`--device cuda|cpu`、`--bpe`、`--ckpt`、`--no-tf32`。

### 6.3 批量参数（`py/infer_batch.py`）

| 参数 | 说明 |
|------|------|
| `--input` | 输入目录或单张图像（必需） |
| `--text` | 文本提示（必需） |
| `--outdir` | 可视化输出目录 |
| `--ext` | 扫描的扩展名（默认 jpg/jpeg/png/bmp/webp） |
| `--save-mask` | 额外导出二值掩码 PNG |
| `--csv` | 把每个目标写入 CSV |

---

## 7. 项目结构

```
SAM3_AGXOrin/
├── README.md
├── requirements.txt
│
├── sh/                           # Shell 入口（只负责环境变量 + 调度）
│   ├── run.sh                    # ⭐ 主入口（自动选择服务或本地）
│   ├── serve.sh                  # ⭐ 常驻服务管理（start/stop/status/log/fore）
│   ├── run_trt.sh                # TensorRT 推理入口
│   ├── setup.sh                  # 一键环境初始化
│   ├── download.sh               # 下载权重与 BPE 词表（断点续传）
│   ├── build_engine_sam3.sh      # ONNX -> TensorRT engine（FP32）
│   └── check_env.sh              # ONNX/TensorRT 环境检查
│
├── py/                           # Python 实现
│   ├── sam3_infer.py             # 推理核心（三提示 + 后处理 + 可视化 + 基准）
│   ├── sam3_client.py            # 客户端：服务优先，自动回退本地
│   ├── sam3_server.py            # 常驻服务（Unix socket + JSON 协议）
│   ├── sam3_trt_infer.py         # TensorRT 版推理（参数同 sam3_infer.py）
│   ├── trt_runner.py             # TRT engine 加载与执行封装
│   ├── infer_batch.py            # 批量文本提示推理
│   ├── export_onnx_sam3.py       # 导出视觉/文本编码器 ONNX
│   ├── verify_onnx_cpu.py        # ONNX 与 PyTorch 精度对齐校验
│   ├── verify_trt_prec.py        # TRT 与 PyTorch 精度对齐校验
│   ├── patch_sam3_triton.py      # 「无 Triton」兼容补丁（Jetson 必需）
│   ├── patch_sam3_rope.py        # RoPE 相关补丁（TRT 路径用）
│   ├── selfcheck.py              # 环境自检
│   ├── probe_sam3.py             # 模型结构探测
│   └── probe_shapes.py           # 张量形状探测
│
├── models/                       # 权重与引擎（需下载/构建）
│   ├── sam3.pt                       # 主权重 3.3 GB
│   ├── sam3_vision_encoder.onnx      # 1.7 GB（可删）
│   ├── sam3_vision_encoder.engine    # 1.7 GB
│   ├── sam3_text_encoder.onnx        # 1.4 GB（可删）
│   └── sam3_text_encoder.engine      # 1.4 GB
│
├── assets/bpe_simple_vocab_16e6.txt.gz    # 文本 tokenizer 词表
├── inputimage/                   # 输入图像（1~8.jpg 为示例）
├── outputimage/                  # 可视化输出
├── outputs/                      # 掩码 / CSV
├── run/                          # 服务运行时文件（socket / pid / log）
├── trt_py/                       # TensorRT Python 绑定的 deb 包与解包目录
└── venv310/                      # Python 虚拟环境（含 NVIDIA 定制 torch）
```

---

## 8. 实现要点

### 8.1 使用官方 Python API

推理接口源自 `examples/sam3_image_predictor_example.ipynb`：

```python
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model(
    bpe_path="assets/bpe_simple_vocab_16e6.txt.gz",
    checkpoint_path="models/sam3.pt",
    load_from_HF=False,          # 只用本地文件，不联网
    enable_segmentation=True,
    device="cuda",
)
processor = Sam3Processor(model, confidence_threshold=0.5, device="cuda")

state = processor.set_image(image)               # 视觉编码（每张图只做一次）
state = processor.set_text_prompt("dog", state)  # 文本提示 -> 掩码

masks  = state["masks"]    # (N, 1, H, W) bool
boxes  = state["boxes"]    # (N, 4)  xyxy 像素坐标
scores = state["scores"]   # (N,)    置信度
```

**关键：视觉编码与提示前向是分离的。** 同一张图换不同提示时，
`set_image()` 的结果可复用，只需重跑提示前向。这也是「图像编码」与
「提示前向」在计时中分开统计的原因。

### 8.2 点提示的实现

官方 `Sam3Processor` 只暴露了 `add_geometric_prompt(box=...)`，
但底层 `Prompt` 类**完整支持点**（`append_points`），几何编码器
`GeometryEncoder._encode_points()` 会真正编码点特征。

本项目在 `py/sam3_infer.py::add_point_prompt()` 中复刻了官方的几何提示流程，
把 `append_boxes` 换成 `append_points`：

```python
pts  = torch.tensor([[[x / iw, y / ih]] for (x, y) in points_px], ...)  # 归一化 [0,1]
lbls = torch.tensor([[int(v)] for v in labels], dtype=torch.bool)
state["geometric_prompt"].append_points(pts, lbls)
return processor._forward_grounding(state)
```

> 注意：`sam3/model/sam3_image.py` 的 `forward()` 里有一句
> `"Warning: Point prompts are ignored in PCS."`，那说的是**训练/批量 PCS 数据管线**，
> **不影响推理路径**（推理走 `forward_grounding`）。

### 8.3 掩码形状陷阱

`state["masks"]` 是 `(N, 1, H, W)`（比常见的 `(N, H, W)` 多一个通道维）。
直接 `vis[masks[i]]` 会报：

```
IndexError: boolean index did not match indexed array along dimension 0
```

`normalize_mask()` 统一 `squeeze()` 到 2D，并在尺寸不符时最近邻缩放回原图。

### 8.4 权重与词表

- **权重**：`facebook/sam3` 的 `sam3.pt`（3.3 GB）。默认从 HuggingFace 下载，
  失败时回退镜像 `1038lab/sam3`。
- **BPE 词表**：`bpe_simple_vocab_16e6.txt.gz`，SAM3 文本编码器需要；
  与 OpenAI CLIP 词表相同，官方仓库未内附，脚本从 `openai/CLIP` 获取。
- 两者均可用 `--bpe` / `--ckpt` 覆盖。

### 8.5 安装踩坑记录

1. **必须 `--no-deps` 安装 sam3**：`pip install sam3` 会把 `torch` 一并拉取，
   用 PyPI 无 CUDA 版本覆盖 Jetson 定制 torch。
   正确：`pip install --no-deps sam3==0.1.4`，再手动装运行依赖。
2. **`numpy` 必须 < 2**：torch 2.8 for Jetson 按 numpy 1.x 编译。
3. **`LD_LIBRARY_PATH` 需含 cuSPARSELt**：
   `venv310/lib/python3.10/site-packages/nvidia/cusparselt/lib`，
   否则 `import torch` 可能失败。`sh/run.sh` 已自动设置。

---

## 9. Jetson 关键适配：Triton 问题

### 9.1 问题

`import sam3` 直接失败：

```
ModuleNotFoundError: No module named 'triton'
```

因为 `sam3/model/edt.py` 在**模块顶层**写死 `import triton`，
而该模块位于 `import sam3` 的必经链上：

```
sam3/__init__.py -> model_builder -> model.sam1_task_predictor
  -> model.sam3_tracker_base -> model.sam3_tracker_utils -> model.edt
```

Triton 官方**无 Jetson (aarch64) wheel**，而 EDT 算子只在 SAM3 的
**视频跟踪**中用到。也就是说：**图像分割完全不需要 Triton，却被它挡住了。**

### 9.2 方案

`py/patch_sam3_triton.py` 把 `edt.py` 的顶层导入改为「可选导入」：

```python
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:          # Jetson 等无 Triton 的平台
    _HAS_TRITON = False
    ...占位对象...
```

补丁**幂等**（重复执行会跳过），首次备份为 `edt.py.orig`。

### 9.3 为什么不用「伪造假 triton 包」

最初尝试在 `site-packages` 放 `triton/` 占位包，结果失败：
`torch/utils/_triton.py::has_triton_package()` 仅用 `import triton` 判断
「本机是否有 Triton」。伪造成功后 torch 误判为「有」，于是
`torch/_inductor/runtime/hints.py` 继续索要 `triton.compiler.compiler`、
`triton.backends.compiler` 等真实实现，越陷越深。

让 sam3 感知**真实**环境（无 Triton）才是正解。补丁后
`has_triton_package()` 返回 `False`，torch 走「无 Triton」分支。

### 9.4 影响范围

| 能力 | Jetson |
|------|--------|
| 文本提示图像分割 | ✅ |
| 框提示图像分割 | ✅ |
| 点提示图像分割 | ✅ |
| 视频目标跟踪（EDT） | ❌（无 Triton） |

`USE_PERFLIB=0`（`sh/run.sh` 已设）关闭同样依赖 Triton 的 `sam3.perflib`。

---

## 10. 性能实测

### 10.1 单次运行（含初始化）

```
[计时] 初始化: 导入三方库 1.7s | 模型加载 14.7s | Processor 0.00s | 初始化总计 16.5s
[计时] 首次推理:   图像编码  1039.3ms | 提示前向   472.7ms | 合计  1512.0ms
[计时] 热身后推理: 图像编码   814.2ms | 提示前向   282.1ms | 合计  1096.3ms
[计时] 端到端(含初始化): 19.0s
```

### 10.2 常驻服务（推荐）

| 阶段 | 耗时 |
|------|------|
| 服务启动（加载模型 + 预热，**仅一次**） | **18 s** |
| 后续每次推理（第 1 / 2 / 3 次实测） | **1.28 s / 1.20 s / 1.20 s** |

**加速约 15 倍**，输出与本地运行完全一致。

### 10.3 各部分耗时占比

| 阶段 | 耗时 | 可复用性 |
|------|------|---------|
| 导入三方库 | ~1.7 s | 常驻后省去 |
| 模型加载 | ~13–15 s | 常驻后省去 |
| 图像编码（ViT-H @1008²） | ~814 ms | 同图多提示可复用 |
| 提示前向 | ~282 ms | 每次提示必做 |

> 首次运行含 CUDA 内核初始化开销，脚本会自动预热一次再统计。

### 10.4 基准命令

```bash
./sh/run.sh --image inputimage/6.jpg --text "shoe" --bench --repeat 5
```

输出分别给出**图像编码**、**提示前向**与**端到端**的均值/中位/极值，以及显存占用。

---

## 11. 资源占用与健康

常驻服务运行 1 小时后的实测（AGX Orin 64GB）：

| 指标 | 数值 | 评价 |
|------|------|------|
| 服务进程 RSS | **5.8 GiB** | 占总量 9.5% |
| 进程内存峰值 (VmHWM) | 7.4 GiB | 推理瞬时高水位 |
| 进程 Swap 使用 | **0 B** | 无内存压力 |
| 系统可用内存 | 44.5 GiB / 61.4 GiB (72%) | 余量充足 |
| 系统 Swap 使用 | **0 B** / 30.7 GiB | 从未换页 |
| CPU 温度 | 47.5 °C | 凉 |
| GPU 温度 | 41.9 °C | 很凉 |
| 结温 (tj) | 47.3 °C | 上限约 105°C，余量巨大 |

**内存构成**：SAM3 权重 ~3.2 GiB + CUDA 上下文/工作区 ~1.5 GiB +
PyTorch/NumPy/Python 运行时 ~1.2 GiB。

> **Jetson 说明**：AGX Orin 是**统一内存架构**，CPU 与 GPU 共享同一块 64GB LPDDR5，
> 因此 `nvidia-smi` 无法单独报告显存（显示 `Not Supported`），
> **进程 RSS 就是 GPU 占用**。

**结论**：运行 1 小时无内存增长（峰值与当前差 1.6 GiB）→ **无泄漏，可长期常驻**；
按当前占用，同时跑 3~4 个服务也无压力。

```bash
./sh/serve.sh status    # 随时复查（含显存占用）
```

---

## 12. 常见问题

**Q1. `No module named 'triton'`**
执行 `./venv310/bin/python py/patch_sam3_triton.py`，或重跑 `./sh/setup.sh`。

**Q2. `No module named 'psutil'` / `'hydra'`**
依赖未装全：`./venv310/bin/python -m pip install psutil hydra-core omegaconf`

**Q3. `import torch` 报找不到 `.so`**
`LD_LIBRARY_PATH` 没设。用 `./sh/run.sh`，或手动
`export LD_LIBRARY_PATH=$PWD/venv310/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH`

**Q4. `./sh/run.sh --help` 提示「参数错误」**
`run.sh` 现在经过客户端，缺少 `--image` 会被校验拦截。
查看参数请用：`./venv310/bin/python py/sam3_infer.py --help`。

**Q5. 点提示出来很多框？**
这是 SAM3 的固有行为（grounding 头输出全部候选）。本项目的点筛选与去嵌套默认
已开启；若想看到全部候选，加 `--keep-all`。

**Q6. 显存不足（OOM）**
调小 `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64`；确认无其他进程占用
（`sudo tegrastats`）；或 `./sh/serve.sh stop` 释放常驻服务；`--device cpu` 兜底（很慢）。

**Q7. 服务启动失败 / `run.sh` 卡住**
`./sh/serve.sh log` 看日志；`./sh/serve.sh stop` 清理后重试。
若 socket 残留但服务已死，客户端会**自动回退本地运行**，不会卡住。

**Q8. 提示 "missing and/or unexpected keys"**
正常。SAM3 图像模型只加载权重中的 `detector.*` 部分（视频跟踪部分忽略），
官方 `_load_checkpoint()` 即如此实现。

**Q9. 下载权重太慢**
`./sh/download.sh` 支持断点续传；也可 `HF_ENDPOINT=https://hf-mirror.com` 走镜像。

**Q10. TRT 为什么没有更快？**
见 [第 5 节](#5-tensorrt-加速)：瓶颈在访存而非 Tensor Core，
且 TRT 首次运行有额外初始化开销。日常建议用 PyTorch 路径。

---

## 13. 许可

模型与代码来自 Meta，遵循其
[SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)。
本项目仅为 Jetson 部署适配。