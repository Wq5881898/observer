# Observer 预先研究与可行性分析

日期：2026-08-29  
状态：技术路线建议，可用于进入 MVP 设计与实现

## 1. 结论摘要

Observer 在普通 Windows 笔记本、内置 RGB 摄像头、完全本地处理的条件下可以实现，适合先做个人使用的 MVP。

可可靠实现的部分包括：在席/离席、明显转头或低头、头肩及上半身相对个人基线的持续偏离、持续手持手机、日程提醒与自动启动、事件级截图、会话汇总和可解释评分。

主要限制是“眼睛始终在屏幕周围”不能按专业眼动仪的标准理解。普通摄像头缺少红外照明和固定几何条件，结果会受头部移动、眼镜反光、光照、摄像头位置、多显示器和个体差异影响。第一版应实现“经过个人校准的屏幕区域注意力分类”，而不是承诺精确注视点。若个人验证结果达不到误报目标，再引入专门的 gaze-estimation 模型。

同样，单目摄像头不能判定唯一或医学意义上的“正确坐姿”。系统应该描述为“相对个人舒适中立基线的持续偏离检测”。NIOSH 指出，即使是中立姿势，长时间保持同一位置也会造成疲劳；因此 Observer 应鼓励姿势变化和休息，而不是强迫用户长时间保持一个僵硬姿势。

## 2. 对参考帖的判断

[参考 LinkedIn 帖子](https://www.linkedin.com/posts/ajantha-devi-vairamani_computervision-opencv-mediapipe-activity-7498279616868933632-JCwI) 展示的 Office Posture Monitor 使用 Python、OpenCV、MediaPipe 和 NumPy，实现了摄像头姿势分析、个人参考校准、肩颈对齐、姿势状态和评分。它证明了 Observer 的姿势监测部分可以快速原型化。

帖子方案可以直接吸收的设计：

- MediaPipe 姿态关键点；
- 启动时的个人参考校准；
- 持续状态、评分和可视反馈；
- 会话历史、告警计数、趋势、截图和休息提醒。

需要改进的部分：

- 帖子保存处理后视频，而 Observer 明确不需要完整录像，应改成纯内存帧流和事件截图；
- 帖子没有解决屏幕区域注视判断，需要 Face Landmarker、虹膜相对位置、头部姿态和屏幕点位校准；
- 绝对角度阈值不能代表所有人的良好姿势，必须相对个人基线；
- 评分必须包含观测质量与置信度，不能把摄像头看不清当作用户违规；
- 必须采用持续时间、平滑、迟滞和冷却时间，避免逐帧抖动造成连续报警。

## 3. 功能可行性矩阵

| 能力 | 可行性 | 预期可靠度 | 条件与边界 |
| --- | --- | --- | --- |
| 摄像头读取与本地实时处理 | 高 | 高 | OpenCV 可直接读取内置摄像头；需处理权限、设备占用和断连 |
| 在席/离席 | 高 | 高 | face/pose 同时丢失并持续一段时间；锁屏或系统睡眠时自动暂停 |
| 明显转头、仰头、低头 | 高 | 中高 | Face Landmarker 的 3D 特征和面部变换矩阵可提供头部方向依据 |
| 眼睛是否在屏幕区域 | 中 | 中 | 必须单用户、单主屏、固定摄像头位置、良好光照并做 5/9 点校准 |
| 精确注视到屏幕具体文字/控件 | 低 | 低到中 | 不应作为 MVP 承诺；专业眼动仪更适合该需求 |
| 头颈肩对齐与上半身倾斜 | 中高 | 中高 | 摄像头应拍到双肩；以个人基线和持续时间判断 |
| 腰椎、骨盆、腿部完整坐姿 | 低 | 低 | 笔记本摄像头通常看不到髋部和下肢，不能可靠判断 |
| 手机出现在画面中 | 高 | 中高 | COCO 物体模型包含 `cell phone`；小目标、遮挡和手机壳会影响结果 |
| 手机被手持并持续使用 | 中高 | 中高 | 手机框需靠近手/手腕；头部/视线朝向可增加置信度，不能只凭低头判断 |
| Agenda、自动启动、提醒 | 高 | 高 | 本地数据库和后台调度即可实现 |
| 事件截图而非连续录像 | 高 | 高 | 帧只驻留内存，事件触发时保存 1–3 张证据图 |
| 可扩展扣分、会话评分与趋势 | 高 | 高（计算层面） | 规则事件写入扣分账本；评分不等同于生产力、学习效果或健康诊断 |

## 4. 推荐技术栈

本节给出结论与关键取舍；摄像头、各视觉模块、UI、调度、存储、运行时、通知和打包的完整候选矩阵见 [完整技术选型对比](technology-selection.md)，当前设备适配和后台资源预算见 [成熟度、硬件与性能评估](maturity-hardware-performance.md)。

### 4.1 MVP 主栈

| 层 | 推荐工具 | 选择原因 |
| --- | --- | --- |
| 语言 | Python 3.11 | CV 原型效率高，MediaPipe、OpenCV、Qt、科学计算和打包生态成熟 |
| 摄像头 | OpenCV `VideoCapture` | 官方 MediaPipe Python 指南也使用 OpenCV 读取摄像头帧；Windows 可优先测试 Media Foundation 后端 |
| 面部/眼部 | MediaPipe Tasks Face Landmarker | 输出 478 个面部关键点、52 个 blendshape 和可选面部变换矩阵，支持异步实时流 |
| 身体姿态 | MediaPipe Tasks Pose Landmarker | 输出图像坐标和 3D world landmarks，支持视频跟踪和异步实时流 |
| 手机物体检测 | MediaPipe EfficientDet-Lite0 int8；保留可替换 `ObjectDetectorBackend` | 默认采用官方轻量预训练模型；只有事件准确率不达标才评估 YOLO11n ONNX/OpenVINO |
| 手持关系 | Pose 手腕关键点；必要时增加 Hand Landmarker | 通过手机框与手腕/手部距离、重叠和时间一致性，区分桌面手机与手持手机 |
| 几何与滤波 | NumPy + OpenCV | 头部 `solvePnP`、角度、归一化距离、回归前处理和图像标注 |
| 桌面 UI | PySide6 | Windows 系统托盘、窗口、声音、定时器和线程模型完整；后续也可支持 macOS/Linux |
| 日程调度 | APScheduler | 支持一次性和重复任务；需要跨重启时使用持久化数据存储 |
| 本地数据 | SQLite | 无服务器、事务可靠，适合任务、会话、事件、校准和聚合指标 |
| 模型运行 | MediaPipe 内置运行时；YOLO 后备方案使用 ONNX Runtime 或 OpenVINO | 默认发布包不引入 PyTorch；预训练模型可在 CPU 上完成推理 |
| 打包 | PyInstaller，先 `onedir` 后 `onefile` | 便于先排查模型和 Qt 资源，再生成单文件或安装包 |
| 测试 | pytest + 录制的非敏感测试片段/合成关键点 | 规则引擎应与摄像头和 UI 解耦，便于确定性测试 |

应使用当前的 MediaPipe Tasks API，而不是依赖旧的 `mp.solutions` 示例。Face 和 Pose 的 `LIVE_STREAM` 模式会异步返回结果，并在推理忙碌时丢弃新帧，正好适合“只处理最新帧、不积压延迟”的实时应用。

### 4.2 视线判断的分阶段路线

第一阶段不引入大型模型。使用以下特征形成一个轻量个人模型：

- 头部 yaw、pitch、roll；
- 双眼虹膜中心在各自眼眶内的归一化 x/y 比例；
- `eyeLook*` blendshape 作为辅助信号，而不是唯一依据；
- 面部在画面中的位置、大小和估计距离；
- 摄像头、屏幕分辨率、缩放和显示器布局指纹。

启动校准显示 5 或 9 个屏幕点。每个点采样约 1 秒，去除眨眼与低置信度帧，使用中位数特征拟合带正则化的线性或二次回归，把上述特征映射到屏幕坐标。校准完成后再显示若干验证点；若误差过大，程序不启动“眼睛偏离”硬告警，只启用头部方向的保守判断并提示重新校准。

如果个人实测仍有过多误报，第二阶段再评估 [L2CS-Net](https://arxiv.org/abs/2203.03339)。论文在 MPIIGaze 和 Gaze360 上报告了不同条件下的角度误差，且官方实现为 MIT 许可；但跨设备、跨用户和真实笔记本环境仍需本地验证，模型权重的分发条款也应单独核对。部署时可考虑转换为 ONNX Runtime，避免让完整 PyTorch 成为桌面应用依赖。

[WebGazer](https://webgazer.cs.brown.edu/) 证明了“摄像头 + 多点校准 + 屏幕坐标回归”的方法可行，但它主要面向浏览器且使用 GPLv3。Observer 是 Python 桌面应用，因此更适合复用其校准思路，而不是把它直接作为 MVP 依赖。

### 4.3 手机检测：MediaPipe EfficientDet 与 YOLO 对比

YOLO 可以用于 Observer，但为了采用成熟、低风险的路线，MVP 不选最新的 YOLO26。Ultralytics 官方把 2024 年发布的 YOLO11 标为当前稳定版本；如果默认检测器失败，后备方案从最小的 `YOLO11n` 开始。两种预训练 COCO 模型都包含 `cell phone`，因此验证阶段不需要训练。

| 对比项 | MediaPipe EfficientDet-Lite0 int8（默认） | YOLO11n（后备，导出 ONNX/OpenVINO） |
| --- | --- | --- |
| 本项目用途 | MVP 手机候选框 | 仅在默认方案不达标时验证；后续便于定制训练 |
| 输入尺寸 | 320×320 | 官方检测模型通常为 640×640 |
| 官方公开指标 | Google 在 Pixel 6 CPU 上报告 int8 平均延迟 29.31 ms | 官方报告 COCO mAP 39.5、2.6M 参数；其 CPU ONNX 基准约 56.1 ms |
| 与当前栈集成 | 与 Face/Pose 同属 MediaPipe Tasks，异步接口一致 | 训练/导出用 Ultralytics；应用内建议用 ONNX Runtime 或 OpenVINO |
| 运行依赖 | TFLite/MediaPipe，较轻 | 不应在发布版强制携带完整 PyTorch；导出后运行时可较轻 |
| 小手机/遮挡 | 通用模型可能漏检，需要实测 | 整体 COCO mAP 不能代表手机类别或手持事件准确率，仍需实测 |
| 自定义训练 | 可用 MediaPipe Model Maker/TFLite 流程，但生态相对有限 | 数据标注、微调、验证、导出链路成熟，适合加入本机视角和相似负样本 |
| 部署格式 | TFLite，MediaPipe 直接加载 | ONNX、OpenVINO、TensorRT、LiteRT 等 |
| 许可证 | MediaPipe 代码与官方示例为 Apache 2.0；发布时仍需核对具体模型文件条款 | Ultralytics 代码和模型默认为 AGPL-3.0；闭源、内部企业或商业产品按其官方说明需要 Enterprise/其他商业许可 |
| 主要优势 | 简单、轻量、与现有管线一致、许可风险较低 | 潜在准确率和定制能力更强，部署格式丰富 |
| 主要劣势 | 对小目标和特殊手机姿态可能不够鲁棒 | 计算与打包更复杂，且许可证会影响未来产品形态 |

表中的延迟来自不同硬件，不能用来宣布谁在本机更快；COCO 总体 mAP 也不能替代 `cell phone` 类别和“手持事件”的实测。

建议把选择分为明确的默认路径和升级路径：

1. 人脸、视线和姿势继续使用 MediaPipe Face/Pose，不用 YOLO Pose 替换，因为 COCO 人体关键点不足以覆盖精细眼睛和面部需求；
2. 手机物体框默认使用 EfficientDet-Lite0 int8；先验证它的事件级准确率和资源占用，不做无必要的模型竞赛；
3. 无论哪一个检测器胜出，都必须与手腕/手部、持续时间和置信度融合，YOLO 不能单独决定扣分；
4. 只有 EfficientDet 达不到验收线时才测试 YOLO11n；只有两个预训练方案都不足时，才考虑收集本项目数据并在有 NVIDIA GPU 的机器或云端微调，重点加入小手机、不同手机壳、遮挡，以及遥控器、计算器和黑色笔记本等负样本。

正式运行绝不同时执行两个物体检测模型。默认方案首先要满足手机持续手持事件 `precision ≥ 0.90、recall ≥ 0.80` 和误扣分要求；若不满足才启动后备评估。YOLO 的 AGPL/Enterprise 许可边界也必须在采用前确认。

为了避免业务层绑定模型，实现统一接口：

```text
ObjectDetectorBackend.detect(frame, timestamp)
    -> [ObjectDetection(class_name, confidence, bbox, model_id, latency_ms)]

Implementations:
    MediaPipeEfficientDetBackend
    YoloOnnxBackend / YoloOpenVinoBackend
```

手机规则只能读取标准化 `ObjectDetection`，不得直接导入 MediaPipe、Ultralytics 或 ONNX Runtime。模型切换后，Agenda、提醒、证据、规则和扣分账本不需要修改。

## 5. 判断规则设计

### 5.1 观测质量门

每个时刻先输出 `observable` 和置信度，再输出用户状态。以下情况不计入违规：

- 摄像头断开或被其他程序占用；
- 光线过暗、强逆光或画面严重模糊；
- 眼镜反光、头发/手遮挡眼睛；
- 面部或双肩不在要求的取景区域；
- 模型置信度低或人脸/姿态结果相互矛盾。

质量问题持续时应发出“调整摄像头/光线”的独立提醒，并降低会话的有效观测覆盖率。

### 5.2 在席状态

- `PRESENT`：face 或 upper-body pose 稳定存在；
- `MAYBE_AWAY`：两者暂时丢失；
- `AWAY`：两者连续丢失超过阈值，且摄像头本身正常；
- 计划休息、手动暂停、锁屏、睡眠不计为离席违规。

### 5.3 屏幕注意力

- 注视预测落在主屏允许区域或用户定义的“屏幕周边容差区”时为 `ON_SCREEN`；
- 短暂眨眼、扫视、思考和查看键盘不报警；
- 只有高置信度的 `OFF_SCREEN` 持续超过阈值才进入提醒；
- 多显示器需要每种布局单独校准，MVP 只支持一个主显示器；
- 看纸质资料是否允许应成为任务规则，而不是系统硬编码。

### 5.4 姿势偏离

校准时记录用户舒适中立姿势 20–30 秒，使用稳健统计量形成个人基线。候选指标包括：

- 双肩连线倾斜；
- 耳/眼中心相对肩中心的横向偏移；
- 头部 pitch、roll 和 yaw；
- 头部大小或面部深度相对基线的变化，用于发现持续前伸/过近；
- 髋部可见时才计算躯干倾角。

距离以肩宽、脸宽等尺度归一化，阈值相对个人基线设置。建议在 2 项以上指标持续异常时才触发姿势提醒，避免把正常伸展或调整坐姿误判为问题。

### 5.5 手机持续手持与过度使用

不能把 `cell phone` 检测框直接等同于“正在玩手机”。建议状态为：

- `PHONE_NOT_SEEN`：没有高置信度手机；
- `PHONE_VISIBLE`：手机在画面中，但没有足够证据说明被手持；
- `PHONE_HELD`：手机框与左/右手腕或手部区域持续接近或重叠；
- `PHONE_USE_SUSPECTED`：在 `PHONE_HELD` 基础上，头部/视线方向也朝向手机区域；
- `PHONE_OVERUSE`：`PHONE_HELD` 或更高状态持续超过规则允许时间；
- `PHONE_UNOBSERVABLE`：手和手机可能在摄像头画面下方或严重遮挡，不能判断。

只有高置信度 `PHONE_HELD`/`PHONE_USE_SUSPECTED` 的连续时间可进入提醒和扣分。手机放在桌上不能扣分；仅检测到低头、手位于桌下或一个矩形物体也不能扣分。手机完全离开画面时，系统只能报告“未观察到手机”，不能反推用户没有使用手机。

Object Detector 不必逐帧运行。无论使用 EfficientDet 还是 YOLO，建议手机检测 2–5 FPS，检测到候选框后临时提高频率，并使用跨帧 IoU/中心距离跟踪。Pose 已经提供手腕关键点；如果验证发现手腕位置不足以区分手持关系，再增加 Hand Landmarker，而不是一开始就承担额外推理开销。

用于试验的初始规则可设为：连续手持 20 秒温和提醒，60 秒开始扣分，120 秒强提醒；手机明确离手、且手部保持可见 3 秒后结束事件。若手机和手一起移出画面，应暂停扣分并标记为 `PHONE_UNOBSERVABLE`，不能当作已经恢复。所有时长、分值和上限必须能按 Agenda 任务配置。例如需要手机完成双因素认证的任务可以设置 30–60 秒宽限，或由用户临时标记一次合理使用。

### 5.6 时间平滑与分级提醒

下面是用于试验的默认值，不是最终产品承诺：

| 事件 | 温和提醒 | 强提醒 | 恢复条件 |
| --- | ---: | ---: | --- |
| 高置信度视线离开屏幕 | 5 秒 | 15 秒 | 连续 2 秒回到允许区 |
| 离席 | 20 秒 | 60 秒 | 稳定检测到本人 2 秒 |
| 姿势持续偏离 | 30 秒 | 60 秒 | 回到个人包络 5 秒 |
| 持续手持手机 | 20 秒 | 120 秒；60 秒起扣分 | 手机明确离手且手部可见 3 秒 |
| 画面无法判断 | 10 秒提示设备/环境 | 不计违规 | 质量恢复 3 秒 |

实现上使用短窗口中位数或指数平滑、进入/退出不同阈值的迟滞、同类告警冷却时间，以及事件合并，避免告警风暴。

## 6. Agenda 与会话流程

状态流程建议为：

`IDLE → CALIBRATION_REQUIRED/READY → MONITORING → WARNING → ALARM → RECOVERED → ENDED`

Agenda 任务至少包含：标题、开始时间、计划时长、重复规则、提醒提前量、观察规则配置、允许的资料区域、手机宽限/禁用策略和休息策略。

到点流程：

1. 系统托盘和声音提醒任务开始；
2. 若摄像头与校准有效，自动建立会话并启动观察；
3. 若设备或校准无效，进入“等待修复”，不生成违规分数；
4. 用户可以暂停、延后、提前结束或标记计划休息；
5. 结束后立即生成分项评分、有效覆盖率、逐笔扣分账本、事件时间线和证据图。

长时间保持同一姿势不应被当作目标。休息规则应可配置；可将每小时短休作为默认建议。CCOHS 建议每在工作站工作一小时休息 5–10 分钟，并尽可能站立或走动。

## 7. 评分模型

评分应叫“会话规则遵守度”，不能叫“学习能力”或“生产力分数”。第一版保存分项指标、扣分账本和原始聚合数据，允许以后调整公式而不重放视频。

建议输出：

- `coverage`：可有效观测时间 / 应观察时间；
- `attention`：高置信度可观测时间内，处于允许屏幕区域的比例；
- `posture`：姿势可观测时间内，处于个人包络的比例；
- `presence`：非计划休息时间内在席的比例；
- `break_compliance`：是否按设定完成休息；
- `base_score`：可配置加权分数，初始建议为 `45% attention + 30% posture + 15% presence + 10% break`；
- `penalty_ledger`：手机及后续新增规则产生的逐笔扣分；
- `overall`：`max(0, base_score - penalty_ledger 合计)`。

当 `coverage < 80%` 时，整体结果显示“证据不足”，而不是给出一个看似精确的低分。每个分数都应能展开看到时长、事件数、阈值、置信度和相关证据。

手机扣分的试验默认值可以是：持续 60 秒时扣 2 分，之后每完整增加 60 秒扣 1 分，单次事件最多扣 5 分、单个会话最多扣 15 分。最终数值应由实际体验调整，不应硬编码在检测器中。

为支持后续其他扣分项，每个规则采用统一定义：

- `rule_id`、名称和版本；
- 是否启用、适用的 Agenda/规则配置；
- 依赖的检测信号和最低置信度；
- 宽限、提醒、扣分开始、恢复、冷却时间；
- 固定、按时长或分段扣分方式，以及单事件/单会话上限；
- 允许条件、临时豁免和计划休息行为；
- 证据保存策略、解释文本和隐私等级；
- 与其他规则重叠时的优先级和冲突组。

规则计算接口只接收标准化 `Observation` 和 `SessionContext`，输出 `RuleSignal`；事件状态机统一负责持续时间、提醒、证据和扣分。这样以后新增扣分项时，只增加检测信号和规则定义，无需改写 Agenda、会话和报告主流程。

为避免重复惩罚，重叠时段必须有主因规则。例如已经成立的手机过度使用事件可以继续记录 `OFF_SCREEN`，但默认只由手机规则扣分，视线规则在同一时段不再重复扣分。

## 8. 隐私与证据存储

默认采用 local-first：不联网、不上传、不做身份识别、不保存连续视频。

摄像头线程只保留最近几秒的内存环形缓冲。一个持续违规事件最多保存三张图：事件成立时、最严重时、恢复时。证据图建议缩放到 640×480 或相近分辨率，并叠加测量值、阈值、时间和置信度；根据用户设置可保存原图、只保存带标注图，或完全禁用图像。

SQLite 建议包含：

- `tasks`：Agenda 任务；
- `sessions`：开始/结束、分项分数、覆盖率、版本；
- `events`：类型、开始/结束、严重度、置信度、测量 JSON；
- `rule_definitions`：规则版本、阈值、扣分方式、上限和冲突策略；
- `penalty_ledger`：规则事件、扣分值、原因、配置快照和是否被豁免；
- `evidence`：文件、哈希、创建时间、删除期限；
- `calibrations`：摄像头/显示器指纹、基线参数、验证误差；
- `settings`：保留期、告警和规则配置。

证据图和数据库默认保留 30 天，可设为 0 天或手动永久保留。删除必须同时删除文件、记录和缩略图。后续发布版应使用操作系统凭据存储保护本地加密密钥，并对数据库与图片做静态加密。若未来用于监控他人，必须先进行当地隐私、劳动和生物特征数据合规评估，并获得明确同意；当前范围仅限本人自我管理。

## 9. 软件架构

```text
Agenda / UI / System tray
          │
          ▼
    Session controller
          │
Camera → latest-frame buffer → Face + Pose + Object/Hand inference
                                  │
                                  ▼
                       Feature / quality extraction
                                  │
                                  ▼
                         Personal calibration
                                  │
                                  ▼
                  State estimator + temporal smoothing
                                  │
                                  ▼
        Rule registry + engine ─→ Alert / evidence selector
                  │                    │
                  └──────────┬─────────┘
                             ▼
               SQLite + penalty ledger + evidence files
                             │
                             ▼
                       Session report
```

UI 必须留在 Qt 主线程；摄像头和推理运行在工作线程。帧队列只保留最新帧，不能让推理落后于实时画面。当前电脑的起始配置为 face 8–10 FPS、pose 3–5 FPS、object 1–2 FPS；界面预览与推理频率解耦。检测到手机候选后可以短时提高 object/hand 频率。具体频率必须通过目标机器基准测试确定，并在系统繁忙或电池模式下降低。

规则引擎只接收带时间戳的特征，不直接依赖 OpenCV 或 MediaPipe，使其可以用合成序列做单元测试。

## 10. 验证方案与 MVP 验收线

不要只看演示画面“好像能跑”。应为目标用户和目标设备录制或现场标注一组场景：正常工作、短暂扫视、持续看左/右/下方、离席、前倾、侧倾、伸展、戴眼镜、低光、手遮脸、视频会议占用摄像头、锁屏/唤醒、查看纸质资料，以及手机放在桌上、短暂拿起手机、持续手持、双手操作、不同手机壳、局部遮挡、拿遥控器/计算器/黑色笔记本等相似物体。

每种场景至少重复 3 次，评价：

- 事件级 precision、recall 和 F1；
- 每小时误报数；
- 从规则成立到告警的延迟；
- 有效观测覆盖率；
- CPU、内存、温度和电池消耗；
- 摄像头恢复、系统休眠和应用重启后的状态一致性。

建议的个人 MVP 验收目标：

- 在席/离席事件 F1 ≥ 0.95；
- 在正常光照、完成校准的单屏环境中，注意力和姿势事件 precision ≥ 0.85；
- 手机持续手持事件 precision ≥ 0.90、recall ≥ 0.80（手机和手位于有效取景范围内）；
- 正常会话误报 ≤ 2 次/小时；
- 告警计时误差的 95 分位 ≤ 1.5 秒；
- 正常条件有效观测覆盖率 ≥ 90%；
- 目标电脑上推理不积压，UI 始终可响应；
- 60 分钟典型工作负载下，Observer 总 CPU 平均 ≤ 15%、95 分位 ≤ 25%；
- 常驻内存稳态 ≤ 500 MB，且一小时增长 ≤ 50 MB；超限时必须自动降低采样率。

这些是项目验收线，不是研究模型的通用准确率承诺。如果最初 MediaPipe 方案无法达到注意力 precision 目标，再进入 L2CS/ONNX 技术升级；若仍无法达到，应放宽为“明显离开屏幕方向”检测，而不是隐藏误差。

## 11. 实施阶段与估算

按一名熟悉 Python 的开发者估算：

1. 技术验证（3–5 天）：摄像头、Face/Pose 异步流、EfficientDet 手机检测、关键特征、资源预算和设备异常；仅当 EfficientDet 不达标时增加 YOLO11n 后备验证；
2. 核心 MVP（9–15 天）：校准、手机/手持关系、可扩展规则状态机、扣分账本、提醒、Agenda、SQLite、证据和会话报告；
3. 个人场景验证（3–5 天）：标注测试、阈值调优、误报和性能修复；
4. 发布准备（3–5 天）：数据删除、恢复测试、PyInstaller、启动项、崩溃恢复和文档；
5. 可选增强视线模型（5–10 天）：L2CS/ONNX、重新校准、精度/资源对比和许可复核。

MVP 总体约 3–6 周；若先做命令行/简单窗口原型，可以在第一周内验证最关键的摄像头、离席、头部方向、姿势偏离和手机识别。

## 12. 主要风险与缓解

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| 眼镜反光、暗光、摄像头分辨率低 | 视线误报 | 质量门、重新校准、允许区扩张、禁止低置信度告警 |
| 笔记本摄像头位置过低或只拍到脸 | 姿势关键点不足 | 校准向导检查双肩可见；不能判断时禁用相应指标 |
| Zoom/Teams 等占用摄像头 | 观察中断 | 明确设备状态，暂停评分，周期性重连，不记录为离席 |
| 多显示器或移动笔记本 | 校准漂移 | 记录显示器和摄像头指纹；布局变化强制重新校准 |
| 手机小、被遮挡或位于画面下方 | 漏检 | 明确有效取景范围、降低采样间隔、必要时升级模型；不可观察时不扣分 |
| 遥控器/计算器等被误认成手机 | 错误扣分 | 类别置信度 + 手持关系 + 跨帧一致性；用相似负样本做个人验证 |
| 阈值过严造成告警疲劳 | 用户弃用 | 分级告警、持续时间、冷却、按任务配置、可解释反馈 |
| 分数被误解为效率或健康结论 | 错误决策 | 命名为规则遵守度，展示分项、覆盖率和证据，不做医学声明 |
| 关键图像泄露 | 隐私风险 | 本地最少化、短保留期、加密、明确删除、无云上传 |
| 打包后模型/Qt 资源缺失 | 发布失败 | 先 onedir、自动冒烟测试，再制作 onefile/安装包 |

## 13. 最终建议

批准项目进入 MVP，采用“Python 3.11 + OpenCV + MediaPipe Tasks Face/Pose/Object Detector（必要时 Hand）+ EfficientDet-Lite0 int8 + PySide6 + SQLite + APScheduler 3.11.x”的本地桌面路线。YOLO11n ONNX/OpenVINO 是默认方案验证失败后的升级路径，不是 MVP 常驻依赖。

第一版的关键产品定义应固定为：

- 监测的是“在席、屏幕区域注意力、相对个人基线的上半身偏离”；
- 手机扣分必须建立在持续、高置信度的手持关系上，不能只凭手机出现或用户低头；
- 评分使用可追溯扣分账本，新增规则通过统一规则接口接入；
- 使用持续事件而非逐帧标签；
- 任何低置信度状态都不能判违规；
- 只保存少量事件证据，不保存完整视频；
- 分数必须带覆盖率且可追溯；
- 首版只支持 Windows、单用户、单主屏；
- L2CS/ONNX 是验证失败后的升级路径，不是 MVP 的前置依赖。

## 14. 主要资料

- [MediaPipe Face Landmarker Python 官方指南](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python)：实时流、478 个面部关键点、blendshape 和面部变换矩阵。
- [MediaPipe Pose Landmarker Python 官方指南](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)：图像坐标、3D world landmarks、实时异步处理。
- [MediaPipe Object Detector Python 官方指南](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector/python)：实时物体检测、类别 allowlist、置信度阈值和边界框。
- [MediaPipe 推荐物体模型与标签](https://developers.google.com/edge/mediapipe/solutions/vision/object_detector)：EfficientDet-Lite0 与 COCO 80 类模型；[官方标签表](https://storage.googleapis.com/mediapipe-tasks/object_detector/labelmap.txt)包含 `cell phone`。
- [MediaPipe Hand Landmarker Python 官方指南](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)：必要时用于更精细的手部关键点和手持关系。
- [Ultralytics YOLO11 官方文档](https://docs.ultralytics.com/models/yolo11)：官方当前稳定版本、模型规模、COCO 指标和 CPU ONNX 参考延迟。
- [Ultralytics 模型导出文档](https://docs.ultralytics.com/modes/export)：ONNX、OpenVINO、TensorRT、LiteRT 等部署格式与量化选项。
- [Ultralytics 许可证说明](https://www.ultralytics.com/license)：AGPL-3.0 与 Enterprise 使用边界；未来闭源或商业发布前必须复核。
- [OpenCV VideoCapture 官方文档](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)：摄像头和视频采集接口。
- [L2CS-Net 论文](https://arxiv.org/abs/2203.03339) 与 [官方实现](https://github.com/Ahmednull/L2CS-Net)：可选的专用视线方向估计路线。
- [WebGazer 项目与论文入口](https://webgazer.cs.brown.edu/)：普通摄像头、多点校准和屏幕坐标回归的参考方案。
- [NIOSH 人体工学风险因素](https://www.cdc.gov/niosh/ergonomics/ergo-programs/risk-factors.html)：中立、别扭和静态姿势以及持续时间因素。
- [NIOSH 居家工作人体工学建议](https://www.cdc.gov/niosh/blogs/2020/working-from-home.html)：坐姿、姿势变化、休息和眼疲劳建议。
- [CCOHS 工作站伸展与休息建议](https://www.ccohs.ca/oshanswers/ergonomics/office/stretching.html)：每小时休息并改变体位。
- [Qt for Python QSystemTrayIcon](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/QSystemTrayIcon.html)：跨平台托盘图标和桌面消息。
- [APScheduler 用户指南](https://apscheduler.readthedocs.io/en/master/userguide.html)：本地任务调度与持久化作业。
- [PyInstaller 官方文档](https://pyinstaller.org/en/stable/)：Python 桌面程序打包。
