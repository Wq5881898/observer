# Observer

Observer 是一个本地优先的专注与坐姿辅助程序。它使用电脑摄像头，在用户设定的学习/工作时段内判断：

- 使用者是否仍在座位前；
- 视线是否持续离开已校准的屏幕区域；
- 头、颈、肩和上半身是否持续偏离个人基线；
- 是否持续检测到清晰可见的手机，并在达到门槛后并入专注违规事件；
- 是否到了 Agenda 中任务的开始时间或休息时间。

系统不会录制整段视频。摄像头帧默认只在内存中处理；只有达到告警条件的事件才保存少量关键图像、测量值和评分依据。

## 当前状态

项目已完成预研、技术地基，阶段 C 已通过并关闭。Python 3.11 隔离环境、DirectShow 摄像头最新帧采集、资源预算监控、MediaPipe Face/Pose、YOLO11n 单类手机检测和 EfficientDet 人物补充探针已经在目标电脑上运行；两轮自然短测、冻结后离席/手机事件链、最终最小盲测及 60 分钟无录像自然长测均已完成。长测运行 `3599.985` 秒、采集 `45,615` 帧且零读取失败，CPU 平均 `12.51%`、P95 `16.79%`、内存峰值 `403.92 MB`、估算增长 `18.61 MB/小时`，均在阶段预算内。实测、失败记录和限制见 [C 收尾核查](docs/phase-c-closeout-audit-20260909.md)，复杂度取舍见 [C 复杂度与止损评审](docs/phase-c-complexity-review-20260910.md)。C阶段曾分别输出注意力、姿态和人体工学信号；D产品当前把注意力纳入时间比例评分，并把姿态与人体工学候选合并为一个零计分“姿态”软提示。

当前产品计分的最新批准基准是[专注违规合并与时间比例评分需求](docs/phase-d-focus-scoring-requirements-20260916.md)：离屏、可见手机和离席保留分项事实，但重叠行为合并提醒，违规秒数按时间并集计算，新会话按实际工作时间比例评分。历史研究和阶段记录中的固定3分/5分方案仅描述当时版本。

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
7. 所有计分信号都通过统一规则注册；重叠信号由合并层计算时间并集，后续增加新信号不得绕过会话、证据和评分主流程。

## 建议的首个交付目标

Windows单用户、单主屏、单摄像头MVP当前包括：个人校准、单次及重复日程、窗口运行时自动开始观察、清晰可见手机检测、可扩展规则、分级提醒、合并事件证据、界面内会话报告和本地数据管理。精确注视点、多显示器、云同步和医学级姿势判断不进入第一版。

## 极简桌面窗口

E1 已接入托盘：关闭窗口后会继续后台运行，可从托盘恢复、暂停/继续或结束会话；完全退出使用窗口底部“退出程序”或托盘“退出”。系统托盘不可用时关闭仍会退出。详见 [E 桌面使用说明](docs/phase-e-runbook.md) 和 [E1 验证记录](docs/phase-e1-result-20260921.md)。

```powershell
uv sync --extra desktop --extra yolo
.\.venv\Scripts\python.exe -m observer product --data-dir data/product-demo gui --demo
```

模拟模式不使用摄像头。真实运行用 `python -m observer product gui`；若存在 `data/calibration/default.json` 会自动启用个人校准，否则界面明确显示仅启用离席与可见手机。`Observer`标题下方提供“专注 / 历史 / 高级计划”三个完整页面；重复计划、计划休息、本次延长和今日摘要集中在高级页。计分违规使用两声短提示、红色横幅和任务栏闪烁；姿态与健康提醒使用一声柔和提示和浅色横幅，不闪烁。详见 [使用说明](docs/phase-d-runbook.md)。

## 开发与验证

F 已生成 Windows onedir 内部候选，含 GUI 与诊断入口、独立用户数据路径、构建来源及文件哈希。候选编号和实际验收状态见 [F 结果报告](docs/phase-f-result-20261010.md)，启动及重新构建步骤见 [F 使用说明](docs/phase-f-runbook.md)。E 的设备恢复及 60 分钟补验保留待测，内部候选尚未正式发布。

已准备供独立 Windows 环境使用的内部 ZIP 验收包，附 Windows PowerShell 5.1/7 入口，不要求 Python 或源码仓库。操作及人工点验见 [转移验收说明](packaging/PORTABLE-VALIDATION.txt)。本机解压往返已验证；项目所有者确认暂时没有独立环境，干净机器验收继续待测。

2026-10-10 已按项目所有者授权发布私有 GitHub 预发布版 [v0.1.0-preview.1](https://github.com/Wq5881898/observer/releases/tag/v0.1.0-preview.1) 并同步源码，版本内容及启动说明见 [发布说明](docs/releases/v0.1.0-preview.1.md)。Release 提供 Windows x64 ZIP、SHA-256 校验文件和构建记录。预发布保留 E/F 待验项，不代表阶段完整签收。

D阶段已于2026-09-21通过并关闭，288项自动验收通过，包括单次及重复任务、会话控制、计划休息、本次延长、离屏豁免、schema v9合并事件、统一姿态事件、时间比例评分、SQLite历史与今日摘要、截图和删除。离屏先发生、手机随后加入的真人场景已确认合并为一个事件，重叠秒数只计算一次。开发版另提供测试取证开关，可在违规候选期间每2秒保存一张复核图片；正式策略为成立的计分事件保存首尾两张、姿态事件最多一张。也可用 `python -m observer product run --demo` 在不打开摄像头的情况下体验；真实模式需安装 `yolo` 依赖并具有冻结模型。具体命令见 [D 阶段使用说明](docs/phase-d-runbook.md)，最终结论见 [D阶段关闭审查](docs/phase-d-closeout-audit-20260921.md)。

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


## 公开源码与更新页面

Observer 自有源码采用 AGPL-3.0-only，条款见 LICENSE；第三方组件保留各自许可，见 NOTICE 与 packaging/REDISTRIBUTION.md。

更新页仅保留当前版本、检查更新、GitHub 发布页和简短状态，无需访问令牌。发布页提供可下载的 Windows 版本；本地开发版本可能领先于已发布版本。

本仓库不包含个人照片、录像、数据库、个人校准和训练数据。第三方对应源码及模型交付范围仍按 packaging/SOURCE-DELIVERY.txt 说明核对，仓库公开不代表旧二进制的材料审核已完成。
