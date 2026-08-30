# Observer 完整技术选型对比

日期：2026-08-29  
目标环境：Windows 笔记本、内置摄像头、单用户、本地处理、CPU 优先

## 1. 选型原则

1. 检测模型只产生事实信号，例如人脸、关键点、手机框；持续时间、提醒和扣分由独立规则引擎负责。
2. 低置信度和超出取景范围必须表示“无法判断”，不能被模型猜测为违规。
3. 模型后端可以替换，Agenda、证据、评分和 UI 不依赖某个模型库。
4. 先用预训练模型做个人环境验证，只有指标不达标才训练自定义模型。
5. 选型同时考虑准确率、误扣分、CPU/内存、离线部署、打包大小和许可证。

## 2. 全模块决策矩阵

| 模块 | 方案 A | 方案 B/C | 当前决定 | 原因 |
| --- | --- | --- | --- | --- |
| 摄像头采集 | OpenCV `VideoCapture` | Qt Multimedia；直接调用 Media Foundation | 选择 OpenCV | 与 NumPy/MediaPipe/ONNX 帧格式衔接最短；先处理权限、占用和重连 |
| 人脸与精细眼部 | MediaPipe Face Landmarker | OpenFace 2；专用人脸模型 | 选择 MediaPipe | 478 个面部点、blendshape、变换矩阵和实时异步流；Python 集成更直接 |
| 屏幕视线 | MediaPipe 特征 + 5/9 点个人回归 | L2CS-Net；OpenFace gaze；WebGazer | 分阶段 | 先做区域判断；不达标再评估 L2CS/ONNX，避免一开始引入重模型 |
| 上半身姿态 | MediaPipe Pose Landmarker | YOLO11 Pose；MoveNet | 选择 MediaPipe | MediaPipe 已覆盖本项目姿态需求；YOLO/MoveNet 默认是 17 点，替换会增加验证与部署成本 |
| 手机物体框 | EfficientDet-Lite0 int8 | YOLO11n ONNX/OpenVINO；SSD MobileNet | 选择 EfficientDet；YOLO11n 仅作升级路径 | EfficientDet 是 MediaPipe 官方推荐轻量模型，320×320 int8、接口统一；仅在事件指标不足时承担 YOLO 的许可和部署成本 |
| 手机是否手持 | Pose 手腕 + 手机框关系 | Hand Landmarker；自定义 `phone-held` 模型 | 先 Pose，必要时 Hand | 手持是物体与人体的关系，不能只看手机类别置信度 |
| 在席/离席 | Face/Pose 存在性 | YOLO `person`；背景变化 | Face/Pose 主判，YOLO 可作辅助 | 已有模型足够；不为一个重复信号增加常驻推理成本 |
| 多人干扰 | Face 数量/人物数量 | YOLO `person` | MVP 简单标记干扰 | 单用户产品不需要身份识别；检测到多人时降低置信度或暂停评分 |
| 低光/模糊/遮挡 | OpenCV 亮度、对比度、Laplacian 与关键点置信度 | 训练图像质量模型 | 选择规则组合 | 成本低、可解释，足够作为观测质量门 |
| 时间与行为事件 | 独立状态机、窗口平滑、迟滞 | 视频动作识别网络 | 选择规则状态机 | 手机过度使用、离席和姿势偏离都依赖时间；更易解释、测试和调整 |
| 扣分扩展 | 规则注册表 + `penalty_ledger` | 在代码中写固定 if/else | 选择规则注册表 | 新增扣分项不修改会话、报告和证据主流程；支持版本、上限、豁免和冲突策略 |
| 桌面 UI | PySide6/Qt Widgets | Electron；Tauri + Web 前端 | 选择 PySide6 | 与 Python CV 同进程/线程体系，托盘、窗口、声音和跨平台能力完整 |
| 提醒 | Qt 托盘 + 应用内浮层 + 声音 | Windows 原生 Toast | MVP 组合提醒 | Qt 官方说明托盘消息可能被系统设置隐藏，因此关键告警不能只依赖托盘气泡 |
| Agenda 调度 | APScheduler + SQLite | Qt `QTimer`；Windows Task Scheduler | 选择 APScheduler | 支持一次性/重复任务和跨重启持久化；Qt 定时器只负责短周期 UI 刷新 |
| 本地存储 | SQLite + 证据文件 | JSON；PostgreSQL | 选择 SQLite | 事务、查询、迁移和备份足够，无需本地服务器；JSON 不适合并发和版本演进 |
| 模型运行时 | MediaPipe/TFLite | ONNX Runtime/OpenVINO；PyTorch | 混合但统一封装 | MediaPipe 模型原生运行；YOLO/L2CS 候选走 ONNX/OpenVINO；发布版不强制携带 PyTorch |
| Windows 加速 | ONNX Runtime CPU | OpenVINO；WinML/DirectML | 基准后决定 | OpenVINO 适合 Intel CPU/GPU/NPU；ONNX Runtime 可按硬件切换 Execution Provider |
| 打包 | PyInstaller `onedir` → `onefile` | `pyside6-deploy`；Nuitka | 先 PyInstaller | onedir 易排查 Qt DLL 和模型资源；稳定后再比较体积、启动速度和源码保护需求 |
| 自动化测试 | pytest + 合成时间序列 + 固定测试片段 | 仅人工摄像头测试 | 选择两者结合 | 规则引擎可确定性测试，摄像头和模型仍需个人真实环境验证 |

## 3. YOLO 在哪些模块适合

### 适合

- 手机、遥控器、书本、食物等可见物体的存在和位置；
- 多人或额外人物进入画面；
- 未来有标注数据时训练本项目专用物体类别；
- 通用模型漏检小手机时，用真实笔记本视角数据微调；
- 导出 ONNX/OpenVINO 后在 CPU、GPU 或部分 NPU 上运行。

### 可以使用，但当前不是首选

- 人体姿态：YOLO11 Pose 默认提供 17 个 COCO 人体点，可完成肩、肘、腕、髋等粗姿态；Observer 已需要 MediaPipe 的面部和姿态能力，因此替换收益有限；
- 在席检测：YOLO `person` 可作为 Face/Pose 均失败时的辅助，但不应为了重复信息常驻另一个高频模型；
- “手机正在使用”：可以训练自定义类别或二级分类器，但仍需持续时间、手持关系和任务豁免，模型不能直接扣分。

### 不适合直接解决

- 精确眼睛注视方向、眨眼或眼睛是否看向屏幕具体区域；
- Agenda、提醒时间、会话开始和结束；
- “超过多少秒算违规”以及扣多少分；
- 低置信度处理、计划休息、双因素认证等合理手机使用豁免；
- 证据保留期限、数据加密、报告和趋势分析。

## 4. 未来扣分项如何选技术

| 扣分项类型 | 例子 | 主要信号 | 推荐接入方式 |
| --- | --- | --- | --- |
| 可见物体 | 手机或以后指定的物品 | YOLO/EfficientDet 物体框 | `ObjectDetectorBackend` + 物体规则 |
| 人体行为 | 持续前倾、侧倾、手部特定位置 | MediaPipe Pose/Hand | 关键点特征 + 时间规则 |
| 面部/眼睛 | 视线离开、持续闭眼 | MediaPipe Face；必要时 gaze 模型 | 面部特征 + 个人校准 + 质量门 |
| 人员状态 | 离席、多人干扰 | Face/Pose；必要时 YOLO person | presence 信号 + 状态机 |
| 电脑活动 | 长时间无键鼠、打开禁止应用 | Windows API/应用事件 | 独立系统适配器，不用摄像头或 YOLO |
| 时间纪律 | 迟到、任务提前结束、休息超时 | Agenda 与系统时钟 | 纯规则，不使用 ML |
| 音频行为 | 未来若确有需求 | 本地音频分类 | 单独评估隐私和权限，不进入 MVP |

新增规则应先归类，再选择最小检测信号。只有现有通用模型无法达到事件级准确率时，才收集和标注数据训练新模型。

## 5. 推理管线建议

```text
Camera frame
    ├─ Face Landmarker ──→ face / iris / head features
    ├─ Pose Landmarker ──→ shoulders / wrists / torso features
    └─ ObjectDetectorBackend
          ├─ MediaPipe EfficientDet
          └─ YOLO11n ONNX/OpenVINO（仅升级路径）

All signals → Observation → Quality gate → Rule engine
            → Event → Alert / Evidence / Penalty ledger
```

各模型采用不同频率：普通模式下 Face 约 8–10 FPS、Pose 约 3–5 FPS、手机物体检测约 1–2 FPS；检测到候选事件时才临时提高相关模型频率。队列只保留最新帧，正常监控时关闭视频预览，不执行模型训练。

## 6. 当前已定与待验证

### 已定

- Python 3.11、OpenCV、MediaPipe Face/Pose/Object Detector、PySide6、SQLite、APScheduler 3.11.x；
- EfficientDet-Lite0 int8 为默认手机检测器，MVP 不训练深度模型；
- 本地处理、事件截图、可解释规则和扣分账本；
- 人脸/姿势与物体检测通过标准化信号隔离；
- 不把低置信度或画面外状态当作违规。

### 待目标电脑验证

- EfficientDet-Lite0 int8 的手机事件准确率和资源开销；只有不达标才测试 YOLO11n ONNX/OpenVINO；
- ONNX Runtime CPU 与 OpenVINO/Windows 硬件加速；
- Pose 手腕是否足够，还是需要额外 Hand Landmarker；
- PyInstaller 发布包的模型资源、启动速度和体积。

最终选型门槛不是演示效果，而是：事件 precision/recall、误扣分次数、有效覆盖率、CPU/内存、温度、电池消耗、打包稳定性和许可证。

## 7. 主要官方资料

- [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)
- [MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)
- [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
- [MediaPipe Object Detector](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector/python)
- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11)：2024 年发布，官方当前标为稳定版本；只作为 EfficientDet 不达标后的升级路径。
- [Ultralytics YOLO Pose](https://docs.ultralytics.com/tasks/pose)：默认 COCO Pose 为 17 个人体关键点。
- [Ultralytics 模型导出](https://docs.ultralytics.com/modes/export)
- [Ultralytics 许可证](https://www.ultralytics.com/license)
- [TensorFlow MoveNet](https://www.tensorflow.org/hub/tutorials/movenet)：17 点人体姿态候选。
- [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)：研究型面部、头姿和 gaze 候选，商业使用需额外核对许可。
- [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
- [Qt for Python QSystemTrayIcon](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QSystemTrayIcon.html)
- [PyInstaller 工作模式](https://pyinstaller.org/en/stable/operating-mode.html)
