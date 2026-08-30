# Observer

Observer 是一个本地优先的专注与坐姿辅助程序。它使用电脑摄像头，在用户设定的学习/工作时段内判断：

- 使用者是否仍在座位前；
- 视线是否持续离开已校准的屏幕区域；
- 头、颈、肩和上半身是否持续偏离个人基线；
- 是否持续手持手机，并在超过允许时间后提醒和扣分；
- 是否到了 Agenda 中任务的开始时间或休息时间。

系统不会录制整段视频。摄像头帧默认只在内存中处理；只有达到告警条件的事件才保存少量关键图像、测量值和评分依据。

## 当前状态

项目已完成预研和第一轮设备技术验证。Python 3.11 隔离环境、DirectShow 摄像头最新帧采集、资源预算监控、MediaPipe Face/Pose 和 EfficientDet-Lite0 分频推理已经在目标电脑上运行通过；实测结果见 [第一阶段技术验证记录](docs/phase-1-technical-validation.md)。统一 Observation、手机/手腕几何融合、持续违规状态机、提醒动作和可扩展扣分账本也已完成，见 [第二阶段规则核心记录](docs/phase-2-rule-engine.md)。

技术边界和整体路线见 [可行性研究](docs/feasibility-study.md)，各模块决策见 [完整技术选型对比](docs/technology-selection.md)，当前电脑适配与运行开销要求见 [成熟度、硬件与性能评估](docs/maturity-hardware-performance.md)。

大阶段、门控和当前状态见 [总体阶段计划](docs/master-plan.md)。阶段 C 按 [阶段 C 基准](docs/phase-c-baseline.md)、[校准与验证协议](docs/phase-c-validation-protocol.md) 和 [设计到真实验证交接](docs/phase-c-validation-handoff.md) 实施；真实数据将同时评价准确率和设计是否过度。

手机检测默认使用 MediaPipe EfficientDet-Lite0 int8。只有它在真实使用场景中达不到手机手持事件验收线时，才评估 YOLO11n ONNX/OpenVINO；正式运行不会同时执行两个物体检测模型。

## 已确定的产品原则

1. 这是个人自我管理工具，不是医疗诊断或员工监控系统。
2. “姿势良好”表示相对个人校准基线的上半身对齐状态，不表示唯一或医学正确的姿势。
3. “看屏幕”表示经校准后落在允许屏幕区域内；普通摄像头不宣称达到专业眼动仪精度。
4. 低光、遮挡、摄像头失败等情况标记为“无法判断”，不能当作违规。
5. 评分必须可解释，并同时显示有效观测覆盖率。
6. 本地处理、最少存储、可配置保留期限和一键删除优先于云端功能。
7. 所有扣分项都通过可配置规则注册；后续增加新规则时不应修改会话、证据和评分主流程。

## 建议的首个交付目标

先完成 Windows 单用户、单主屏、单摄像头 MVP：个人校准、任务日程、自动开始观察、手机持续手持检测、可扩展扣分规则、分级提醒、事件证据、会话报告和本地数据管理。精确注视点、多显示器、云同步和医学级姿势判断不进入第一版。

## 开发与验证

```powershell
uv sync --python 3.11
.\.venv\Scripts\python.exe -m observer doctor
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m observer camera-smoke --seconds 15
.\.venv\Scripts\python.exe -m observer vision-smoke --seconds 30
```

`camera-smoke` 和 `vision-smoke` 都不显示预览、不保存图像或视频，只输出取帧、模型延迟和进程资源统计。模型二进制来自 Google MediaPipe 官方存储，来源和 SHA-256 记录在 `models/manifest.json`；二进制本身不提交到版本库。
