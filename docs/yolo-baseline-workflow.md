# Observer YOLO 基础评估工作流

## 1. 目的与边界

本工作流只建立 YOLO 的可重复评估基础，不直接替换当前手机检测器，也不修改 B01～B13 基线、手机规则、持续事件线、证据策略或正式参数。

当前正式检测器仍为 MediaPipe Tasks + EfficientDet Lite0 int8，正式参数保持：

```text
phone_score_threshold = 0.25
phone_retain_seconds = 2.0
phone_center_distance = 0.18
```

YOLO 先作为独立的候选检测后端进行离线对照。只有通过离线准确率、误报、资源和许可检查后，才允许进入真人 tuning；在没有明确批准前，YOLO 不进入正式运行和 holdout。

### 当前实施状态（2026-09-06 审计后）

- Y0：已保留当前 EfficientDet 正式路径和原有测试目录，未更改正式阈值。
- Y1：已完成统一 `ObjectDetectorBackend`、EfficientDet 适配器和标准化结果。
- Y2：YOLO11n ONNX 基础模型和 ONNX Runtime 1.29.0 已就绪，
  `yolo-doctor` 返回 `ready: true`。
- Y3：`yolo-replay` 已用于 P2 独立录像重放；live 单类别名称
  `cell_phone` 的接入错误已修复。
- Y4：已比较 0.25 与 0.15。降低阈值只能明显改善顶部边缘，不能解决左右
  边缘漏检，并开始引入 recovery 误报。
- Y7：已完成一次 96 张小样本微调。模型能学习 P0/P1，但未通过使用另一种
  手机外观的 P2 独立测试，暂不进入正式路径。
- 回归：143 项自动化测试通过。
- 详细审计：`docs/yolo-phone-model-audit-20260906.md`。

## 2. 工作原则

1. 不在原有基线上直接改参数或替换模型。
2. 先复用已经保存的 P0/P1 边缘手机录像，尽量不要求测试者重复同一动作。
3. EfficientDet 和 YOLO 必须处理同一帧、同一时间段、同一真值标签。
4. 规则层只读取统一的标准化检测结果，不绑定具体模型。
5. 预训练模型优先；没有证据证明预训练 YOLO 不足前，不开始自定义训练。
6. 任何候选优化必须同时检查正例召回和无手机负例误报。
7. 记录模型版本、文件 SHA-256、输入尺寸、阈值、运行时和硬件环境。

## 3. 阶段划分

### Y0：冻结现状

记录当前 EfficientDet 状态：

- 当前 Git commit 和工作树状态；
- `models/manifest.json` 中 EfficientDet 文件和 SHA-256；
- P0/P1 批次目录、报告、录像和证据路径；
- 当前正式参数和当前已知限制；
- 自动化测试基线数量及结果。

Y0 的输出是只读基线清单。不得因为建立 YOLO 目录而修改这些文件。

### Y1：建立统一检测器接口

在应用层定义统一的目标检测结果格式：

```text
ObjectDetection(
    class_name,
    confidence,
    bounding_box,
    model_id,
    latency_ms,
)
```

实现两个后端：

```text
MediaPipeEfficientDetBackend  # 默认，行为保持不变
YoloOnnxBackend               # 仅用于离线诊断
```

手机规则、候选跟踪、证据和报告只能读取标准化结果。Y1 完成后，默认 EfficientDet 的输出和现有测试必须保持不变。

当前代码已完成 Y1 的第一步：`src/observer/object_detection.py` 提供统一
`ObjectDetectorBackend` 接口，默认 EfficientDet 通过
`MediaPipeEfficientDetBackend` 适配器接入；`YoloOnnxBackend` 使用可选的
ONNX Runtime CPU 推理，不会改变默认正式运行路径。可用下面命令检查 Y2
是否具备测试条件：

```powershell
uv sync --extra yolo
.\\.venv\\Scripts\\python.exe -m observer yolo-doctor --model models/yolo/yolo11n.onnx
```

如果本机 `uv` 无法启动，也可以只在当前虚拟环境安装 CPU 运行时：

```powershell
.\\.venv\\Scripts\\python.exe -m pip install "onnxruntime>=1.18,<2"
```

只有 `yolo-doctor` 输出 `"ready": true` 后，才进入模型下载/校验和 Y3
离线回放。模型文件必须单独放在 `models/yolo/`，不能覆盖 EfficientDet。

### Y2：准备 YOLO 预训练模型

首个对照模型使用轻量 YOLO11n 预训练 COCO 模型，优先导出为 ONNX；运行时优先考虑 ONNX Runtime 或 OpenVINO，不把完整 PyTorch 放入 Observer 发布运行时。

需要记录：

- 模型版本和导出参数；
- 输入尺寸和 letterbox/缩放方式；
- `cell phone` 类别映射；
- 模型文件 SHA-256；
- 模型来源和许可证说明；
- CPU 推理依赖和版本。

模型文件放入独立目录，例如：

```text
models/yolo/
  yolo11n.onnx
  manifest.json
```

不得覆盖 `models/efficientdet_lite0_int8.tflite`。

### Y3：离线回放对照

首先使用现有 P0/P1 视频，不启动摄像头，不要求用户重复测试。对每个视频分别运行 EfficientDet 和 YOLO，输出独立目录：

```text
data/model-eval/efficientdet-p0/
data/model-eval/yolo11n-p0/
data/model-eval/efficientdet-p1/
data/model-eval/yolo11n-p1/
```

每个结果至少保存：

- 逐帧候选框和置信度；
- 首次检出时间；
- 漏检间隔；
- 最长连续检测时间；
- 每个场景的 recall、precision、F1；
- 误检框位置分布；
- 每帧推理耗时；
- 运行时错误和无效帧统计。

### Y4：模型参数小矩阵

只有 Y3 后才做阈值对照。对两个模型分别使用：

```text
0.25  保守基线
0.15  中等召回
0.10  低阈值诊断
```

YOLO 的候选保留时间和中心距离先沿用 Observer 当前规则，避免把模型差异和跟踪器差异混在一起。只有发现 YOLO 已经产生稳定候选、但跨帧抖动明显时，才单独测试 retain/center 参数。

### Y5：资源与部署检查

在目标电脑上记录：

- CPU 平均和 P95；
- 内存峰值和增长；
- 单帧推理延迟 P50/P95；
- 摄像头帧率是否下降；
- 与 Face/Pose 同时运行时是否超预算；
- ONNX Runtime/OpenVINO 打包大小和启动时间。

模型不能只因为准确率较高就采用；如果它明显影响用户正常工作，也不能进入正式方案。

### Y6：进入真人 tuning 的门槛

仅当 YOLO 在离线对照中同时满足以下条件，才进入新的真人 tuning：

```text
边缘手机正例 recall >= 0.90
持续检测能够达到 10 秒事件线
无手机 recovery 误报 <= 0.10
误检最长连续时间 < 2.5 秒
资源开销满足 Observer 当前预算
许可证和发布方式可接受
```

真人 tuning 仍使用新的独立场景和独立输出目录，不修改原有正式基线。

### Y7：只有必要时才微调

如果 EfficientDet 和预训练 YOLO 都不能通过边缘手机要求，才考虑使用项目数据对 YOLO 做小规模微调。训练数据必须包含：

- 正下方、左下、右下和上方边缘；
- 不同手机露出比例；
- 手机壳、光照和遮挡变化；
- 遥控器、计算器和相似背景负样本；
- 与真实部署摄像头相同的取景条件。

训练集、验证集和最终 holdout 必须按会话或场景隔离，不能把同一段视频的相邻帧同时放进训练和 holdout。

## 4. 测试场景

首轮只使用独立的手机边缘场景：

```text
phone_edge_e01_bottom_center
phone_edge_e01_bottom_center_recovery
phone_edge_e02_bottom_left
phone_edge_e02_bottom_left_recovery
phone_edge_e03_bottom_right
phone_edge_e03_bottom_right_recovery
phone_edge_e04_top_edge
phone_edge_e04_top_edge_recovery
```

E02/E03 必须以“画面左/画面右”记录，避免摄像头镜像和测试者身体左右造成标签混淆。

## 5. 决策规则

| 结果 | 决策 |
| --- | --- |
| YOLO 明显提高边缘 recall，且负例误报和资源均合格 | 进入独立真人 tuning |
| YOLO 提高 recall，但 recovery 误报超标 | 不采用，先分析阈值、输入预处理或负样本 |
| YOLO 与 EfficientDet 都在同一边缘位置失败 | 进入数据/输入预处理专项，不继续盲目调阈值 |
| YOLO 准确率较高但资源或许可不合格 | 不进入发布方案 |
| 只有定制训练后才达到要求 | 单独建立训练数据和模型版本，不改当前基线 |

## 6. 交接要求

每次模型评估都必须交接以下内容：

1. 模型文件和 SHA-256；
2. 运行时、输入尺寸和阈值；
3. 对应视频和报告目录；
4. EfficientDet/YOLO 对照表；
5. 误报与漏检证据图；
6. CPU、内存和延迟数据；
7. 是否允许进入真人 tuning 的明确结论；
8. 是否改变正式模型的明确决定。

在 Y6 通过前，正式运行继续使用 EfficientDet；在 Y7 训练完成并经过独立 holdout 前，不得宣称 YOLO 已达到生产验收。
