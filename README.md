# Observer

Observer 是一个本地优先的专注与坐姿辅助程序。它使用电脑摄像头，在用户设定的学习/工作时段内判断：

- 使用者是否仍在座位前；
- 视线是否持续离开已校准的屏幕区域；
- 头、颈、肩和上半身是否持续偏离个人基线；
- 是否持续手持手机，并在超过允许时间后提醒和扣分；
- 是否到了 Agenda 中任务的开始时间或休息时间。

系统不会录制整段视频。摄像头帧默认只在内存中处理；只有达到告警条件的事件才保存少量关键图像、测量值和评分依据。

## 当前状态

项目已完成预研、技术地基，阶段 C 已通过并关闭。Python 3.11 隔离环境、DirectShow 摄像头最新帧采集、资源预算监控、MediaPipe Face/Pose、YOLO11n 单类手机检测和 EfficientDet 人物补充探针已经在目标电脑上运行；两轮自然短测、冻结后离席/手机事件链、最终最小盲测及 60 分钟无录像自然长测均已完成。长测运行 `3599.985` 秒、采集 `45,615` 帧且零读取失败，CPU 平均 `12.51%`、P95 `16.79%`、内存峰值 `403.92 MB`、估算增长 `18.61 MB/小时`，均在阶段预算内。实测、失败记录和限制见 [C 收尾核查](docs/phase-c-closeout-audit-20260909.md)，复杂度取舍见 [C 复杂度与止损评审](docs/phase-c-complexity-review-20260910.md)。注意力、姿态和人体工学只作为软提醒，不进入效率扣分。

技术边界和整体路线见 [可行性研究](docs/feasibility-study.md)，各模块决策见 [完整技术选型对比](docs/technology-selection.md)，当前电脑适配与运行开销要求见 [成熟度、硬件与性能评估](docs/maturity-hardware-performance.md)。

大阶段、门控和当前状态见 [总体阶段计划](docs/master-plan.md)。阶段 C 按 [阶段 C 基准](docs/phase-c-baseline.md)、[校准与验证协议](docs/phase-c-validation-protocol.md)、[真实验证运行手册](docs/phase-c-runbook.md) 和 [设计到真实验证交接](docs/phase-c-validation-handoff.md) 实施；真实数据将同时评价准确率和设计是否过度。

阶段 C 冻结候选使用本地 YOLO11n ONNX 单类模型检测清晰可见的手机，并以低频 EfficientDet `person` 探针保留遮脸在席能力。极低露边或严重遮挡手机允许漏检或无法判断，不为这些边界继续扩大训练。冻结模型身份和 SHA-256 见 [模型目录说明](models/yolo/README.md)。

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

## 极简桌面窗口

```powershell
uv sync --extra desktop --extra yolo
.\.venv\Scripts\python.exe -m observer product --data-dir data/product-demo gui --demo
```

模拟模式不使用摄像头。真实运行用 `python -m observer product gui`；若存在 `data/calibration/default.json` 会自动启用个人校准，否则界面明确显示仅启用离席与可见手机。任务名、时长、开始时间、暂停/结束和历史摘要都在一个窗口内；软提醒会发出系统提示音并闪烁任务栏。详见 [使用说明](docs/phase-d-runbook.md)。

## 开发与验证

D 阶段已打通可运行的极简垂直切片：单次任务、会话开始/暂停/结束、SQLite 历史摘要、截图开关与记录删除。它用于验证整体方向，不代表D主体或阶段接近完成。也可用 `python -m observer product run --demo` 在不打开摄像头的情况下体验；真实模式需安装 `yolo` 依赖并具有冻结模型。具体命令见 [D 阶段使用说明](docs/phase-d-runbook.md)，当前进度和后续建设见 [D 实施检查点](docs/phase-d-implementation-checkpoint-20260910.md)。

```powershell
uv sync --python 3.11 --extra yolo
.\.venv\Scripts\python.exe -m observer doctor
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m observer camera-smoke --seconds 15
.\.venv\Scripts\python.exe -m observer vision-smoke --seconds 30
.\.venv\Scripts\python.exe -m observer calibrate --output data/calibration/default.json
.\.venv\Scripts\python.exe -m observer validation-scenarios
```

`camera-smoke` 和 `vision-smoke` 都不显示预览、不保存图像或视频，只输出取帧、模型延迟和进程资源统计。模型二进制来自 Google MediaPipe 官方存储，来源和 SHA-256 记录在 `models/manifest.json`；二进制本身不提交到版本库。

`calibrate` 会打开临时本地窗口，引导完成中立姿势和 9 点注意力校准。校准档案、派生数值和真实验证批次都写入 Git 已忽略的 `data/`；不保存连续视频或完整 landmarks。验证默认只为误判保存一张本地关键帧，并写入 7 天到期索引；可用 `validate-live --no-error-evidence` 完全关闭。完整命令和数据交接见 [阶段 C 运行手册](docs/phase-c-runbook.md)。
