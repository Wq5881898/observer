# Observer 成熟度、硬件与性能评估

日期：2026-08-29  
评估对象：当前 Lenovo `20T1S1LU00` Windows 笔记本

## 1. 结论

Observer 的主技术组合属于成熟的本地计算机视觉桌面应用路线，不需要发明新算法，也不需要在 MVP 阶段训练深度神经网络。更准确地说，这类产品没有唯一的“标准整套框架”，但各层采用的都是长期使用、资料充分、可离线部署的常见组件：OpenCV 负责摄像头、MediaPipe 负责人脸/姿态/轻量物体检测、Qt 负责桌面界面、SQLite 负责本地数据、APScheduler 负责任务调度，持续时间和扣分使用普通状态机。

当前电脑可以完成 MVP 开发、测试和 CPU 本地运行。它不适合实际的 YOLO 深度训练，也不适合让多个视觉模型以 1080p/30 FPS 同时满速运行。MVP 不需要这些能力；通过低分辨率、分频推理、只保留最新帧和自适应降载，可以把 Observer 控制在后台辅助程序应有的开销范围。

## 2. 成熟度审视

| 组件/方法 | 成熟度判断 | Observer 中的决定 |
| --- | --- | --- |
| Python 3.11 | 成熟、依赖兼容面稳定 | 固定项目版本，单独虚拟环境，不追逐系统最新 Python |
| OpenCV `VideoCapture` | 摄像头与图像处理的常用方案 | 采用；Windows 上验证 Media Foundation 后端与断线重连 |
| MediaPipe Face/Pose Tasks | 常见的端侧人脸和人体关键点方案，官方支持摄像头实时流 | 采用预训练模型；不自行训练 |
| MediaPipe EfficientDet-Lite0 int8 | 官方推荐的轻量 320×320 COCO 检测模型 | 作为手机框默认检测器，只允许 `cell phone` 类别输出 |
| YOLO11n | Ultralytics 官方当前标为稳定版本；训练、导出和部署链路成熟 | 仅在 EfficientDet 无法通过事件验收时评估；不用更新的 YOLO26 做 MVP；采用前复核 AGPL/Enterprise 许可 |
| PySide6 / Qt Widgets | 成熟桌面 UI 体系 | 采用；关键提醒使用应用内浮层和声音，不能只依赖可能被系统隐藏的托盘消息 |
| SQLite | 非常成熟的嵌入式事务数据库 | 采用；存任务、事件、评分和配置，不存连续视频 |
| APScheduler 3.11.x | 稳定的 Python 任务调度分支 | 采用并锁定 3.11.x；不采用仍处于预发布阶段的 4.x 分支 |
| PyInstaller `onedir` | 常见的 Python Windows 打包方案 | 先采用；稳定后再评估安装包和 `onefile` |
| 状态机、滑动窗口、迟滞 | 工业事件检测中的常规工程方法 | 采用；模型只提供信号，规则才决定提醒和扣分 |

以下内容不进入 MVP：L2CS-Net/OpenFace 等专用 gaze 模型、自定义动作识别网络、连续视频分析训练、YOLO Pose 替换 MediaPipe、云端推理、身份识别。它们只有在预训练主路线无法满足已经定义的验收指标时才重新评估。

这不是“零验证”。摄像头角度、光线和手机大小高度依赖个人环境，因此仍需做 3–5 天的工程验证和阈值校准；这是产品集成测试，不是算法研究。

## 3. 当前电脑检查结果

| 项目 | 当前电脑 | 判断 |
| --- | --- | --- |
| CPU | Intel Core i7-10610U，4 核 8 线程，15 W 移动处理器，支持 AVX2 | 可以做低频 CPU 推理；不应持续占满所有核心 |
| 内存 | 15.8 GB | 开发和运行足够；检查时仅约 3.9 GB 可用，说明需要严格限制常驻内存并关闭不必要的开发进程 |
| GPU | Intel UHD Graphics，共享显存，无 NVIDIA CUDA 独显 | 运行不依赖 GPU；可后测 OpenVINO，但不把它作为前置条件；不适合本地 YOLO 训练 |
| 摄像头 | Integrated Camera，设备状态正常 | 满足输入条件；仍需实际测试分辨率、暗光、帧率和被会议软件占用时的行为 |
| 磁盘 | C 盘约 237.7 GB，总空闲约 86.7 GB | 代码、模型、数据库和事件截图足够；禁止连续录像并设置证据配额/保留期 |
| 系统 | Windows 11 Pro 64 位 | 符合首版目标平台 |
| Python | 当前仅发现 Python 3.13.5 | 不作为保守项目环境；应并行安装 64 位 Python 3.11，建立项目专用 `.venv`，不替换系统 Python |

适配结论：

- UI、Agenda、SQLite、规则引擎和校准：完全足够；
- MediaPipe 预训练模型本地开发与推理：足够，但必须限频；
- 5/9 点个人校准中的线性/岭回归：只是小规模数值拟合，CPU 可在很短时间内完成，不等同于神经网络训练；
- 多模型高帧率演示：不作为产品模式；
- 自定义 YOLO 深度训练：当前硬件不合适。若后期确实需要，使用带 NVIDIA CUDA GPU 的另一台机器或按时租用云 GPU；以 8 GB VRAM 作为小模型试验起点，12 GB 以上更从容。训练产物再导出到当前电脑做 CPU 推理。

## 4. 低开销运行设计

Observer 判断的是持续数秒的行为，不需要像视频会议一样处理每一帧。默认运行档位如下：

| 模块 | 默认策略 |
| --- | --- |
| 摄像头 | 640×480，采集 10–15 FPS；只保存一个最新帧，旧帧直接丢弃 |
| Face Landmarker | 8–10 FPS，`num_faces=1` |
| Pose Landmarker | 3–5 FPS，单人模式，不启用分割掩码 |
| 手机检测 | 平时 1–2 FPS；出现候选后短暂提高到 4–5 FPS 以确认持续性 |
| Hand Landmarker | 默认不加载；只有“手机框靠近手腕但关系不确定”时短时启用 |
| 视频预览 | 正常监控时关闭，只在校准、设置和诊断时显示 |
| 证据 | 帧在内存中处理；事件成立时写入少量图片，不写连续视频 |
| 线程/队列 | 推理线程限制为 2–3 个，不建立帧积压队列；进程使用低于普通应用的优先级 |

还需要三个自适应档位：

1. `normal`：使用上表频率；
2. `busy/battery`：Face 降到约 4–6 FPS、Pose 约 2 FPS、手机约 0.5–1 FPS；
3. `event-confirmation`：只短时提高相关模型频率，事件确认后恢复。

当系统 CPU/内存压力较高、电脑进入电池模式、窗口锁定、睡眠，或摄像头被 Teams/Zoom 等占用时，Observer 应自动降载或暂停评分，并把状态记为“无法观测”，不能因此扣分。

## 5. 首版性能验收预算

以下是当前电脑上的发布门槛，不是尚未实测的性能承诺。必须在 60 分钟典型工作负载下采样，并分别测试接电和电池模式：

| 指标 | 目标门槛 |
| --- | --- |
| Observer 总 CPU | 普通模式平均不超过 15%，95 分位不超过 25%（Windows 全机 0–100% 口径） |
| 常驻内存 | 稳态不超过 500 MB；一小时增长不超过 50 MB |
| 推理延迟 | 不允许帧队列积压；每个模块始终处理最新帧 |
| UI 响应 | Agenda、暂停和结束操作不被推理阻塞，交互响应目标小于 100 ms |
| 磁盘写入 | 无连续视频；只有配置、事件元数据和受配额限制的证据图片 |
| 降载行为 | 超出 CPU/内存预算后自动降低采样率，不能与用户前台应用争夺全部核心 |

若默认组合达不到预算，优化顺序固定为：关闭常驻预览 → 降低 Pose/物体检测频率 → 缩小/裁剪输入区域 → 限制线程 → 评估 OpenVINO。不能为了保住高帧率而提高后台资源占用。若仍不达标，先减少非核心规则，而不是引入更新、更复杂的模型。

## 6. 推荐的最终 MVP 组合

```text
Python 3.11
  ├─ OpenCV：640×480 摄像头和轻量图像质量检查
  ├─ MediaPipe Face Landmarker：面部、虹膜和头部特征
  ├─ MediaPipe Pose Landmarker：肩、躯干和手腕特征
  ├─ MediaPipe Object Detector + EfficientDet-Lite0 int8：低频手机框
  ├─ NumPy / 普通回归与状态机：个人校准、持续时间、迟滞和扣分
  ├─ PySide6：桌面界面、托盘、浮层和声音
  ├─ APScheduler 3.11.x：Agenda
  └─ SQLite：任务、会话、事件、评分和证据索引
```

YOLO11n ONNX/OpenVINO、Hand Landmarker 和专用 gaze 模型都是按验收失败逐级启用的备选项。这个结构把创新控制在产品规则与用户体验上，而不是押注未经验证的新视觉技术。

## 7. 依据

- [MediaPipe Object Detector Python 指南](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector/python)：支持 OpenCV 摄像头帧、异步实时流、类别 allowlist；推理忙时会忽略新帧，适合最新帧策略。
- [MediaPipe Object Detector 模型说明](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector)：官方推荐 EfficientDet-Lite0，含 int8 版本和端侧参考延迟。
- [Ultralytics YOLO11 官方文档](https://docs.ultralytics.com/models/yolo11)：YOLO11 被标为当前稳定版本，YOLO11n 的参数、COCO 指标和 CPU ONNX 参考延迟。
- [Ultralytics 许可证](https://www.ultralytics.com/license)：AGPL-3.0 与 Enterprise 使用边界。
- [Intel i7-10610U 官方规格](https://www.intel.com/content/www/us/en/products/sku/201896/intel-core-i710610u-processor-8m-cache-up-to-4-90-ghz/specifications.html)：4 核 8 线程、15 W、Intel UHD 与 AVX2。
- [APScheduler PyPI](https://pypi.org/project/APScheduler/)：稳定版与 Python 支持状态。
- [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)：CPU、OpenVINO、DirectML 等可替换运行后端。
