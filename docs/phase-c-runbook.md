# Observer 阶段 C 真实校准与验证运行手册

版本：0.1
日期：2026-08-30
状态：实现交接完成；等待项目所有者参与执行

正式收尾测试不再直接穷举本手册中的所有早期 tuning 场景。测试优先级、两个独立循环、盲测要求和停止规则见 `docs/phase-c-formal-test-plan.md`。

## 1. 这一步为什么必须有人参与

自动化测试可以证明数据契约、数学计算、版本保护、隐私默认值和状态融合没有明显回归，但不能替代“本人真实看屏幕、倾斜、离座和手持手机”的人工真值。阶段 C 只有在真实 tuning、冻结后的全新 holdout 和 60 分钟正常使用长测完成后才能判断通过。

全程本地运行，不录制视频。校准与验证数值位于 Git 忽略的 `data/`。验证默认只为预测错误保留每 trial 一张关键帧；如果不希望保存任何图像，在 `validate-live` 命令追加 `--no-error-evidence`。

## 2. 测试前固定环境

开始前保持以下条件不变：ThinkPad 内置主屏与内置摄像头、日常座椅高度、正常桌面光照、日常眼镜状态、同一 Windows 电源模式。打开摄像头隐私挡板，退出占用摄像头的软件，并让完整脸部和双眼进入画面。双肩和手腕不是日常运行的强制取景要求；它们不可见时，相应的完整上身姿态或手机持握子项会单独不可观测，但不得阻塞在席、注意力和 face-only 头部姿态。

若摄像头无法打开，程序应明确提示设备可能被其他应用占用、被 Windows 隐私设置禁用或已断开。部分 Windows 驱动在被占用时仍允许打开设备句柄，却只返回黑色或无效帧；此时程序会保留“无可用图像”提示并等待确认，而不是黑屏后无理由退出。先关闭相机、会议软件、浏览器摄像头页面或其他监控进程，再重新运行；不要在错误窗口上继续校准。

在项目目录执行：

```powershell
.\.venv\Scripts\python.exe -m observer doctor
.\.venv\Scripts\python.exe -m observer camera-smoke --seconds 15
.\.venv\Scripts\python.exe -m observer vision-smoke --seconds 30 --models-dir models
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

前三项失败时不要进入校准。冒烟测试不保存画面；没有检测到 Face/Pose 时先检查取景和隐私挡板。

运行时固定使用 MediaPipe `0.10.21`。首次真人校准发现 `0.10.35` 会产生未文档化的 Clearcut 网络连接尝试，且官方没有提供受支持的关闭开关；为了维持本地优先边界，Observer 回退到已验证无该行为的 `0.10.21`，不以隐藏日志代替阻止连接。

目标设备真人 tuning 发现 MediaPipe FaceLandmarker 默认 0.5 检出/存在/跟踪置信门只得到约 76.1% Face positive，并使三批注意力覆盖率降至 25.8%～41.4%。默认配置现使用官方选项的 0.35；对照三批注意力覆盖率均为 100%。该值仍须通过空座和相似物体场景验证假 Face 风险，不能因为正常人脸召回提高就跳过 negative trials。

## 3. 建立个人校准档案

```powershell
.\.venv\Scripts\python.exe -m observer calibrate --models-dir models --output data/calibration/default.json
.\.venv\Scripts\python.exe -m observer profile-check data/calibration/default.json
```

校准约需 2～3 分钟。置顶窗口先显示镜像预览，坐好并确认头顶、双耳、双眼和双肩完整入镜后，单击预览画面或按空格开始；提供鼠标和键盘两条路径是为了避免 Windows 窗口焦点竞争。前半段保持可持续的中立坐姿，后半段依次注视 17 个拟合点和 8 个未参与拟合的内部验证点。红色 `READY` 点只用于移动视线，不采样；点变绿并显示 `RECORDING` 后才采样，不需要逐点点击。每点至少需要 7 个有效帧，并先取特征中位数再用于拟合或验证；归一化验证 RMSE 必须不高于 0.18，否则档案不保存。按 `Esc` 可安全取消。成功后产生：

眼睛睁开程度只用于剔除眨眼和不可观测帧，不参与注视坐标回归；真实校准发现把眼睑开合量作为位置输入会放大疲劳、眨眼造成的漂移。

运行时对最近 7 个有效 Face 结果取特征中位数：质量或人脸丢失时立即清空，单次自然眨眼只让当帧输出 `unknown`，连续 3 次眼部不可观测才清空。窗口尚未填满时输出带原因的 `unknown`。在默认 8 FPS 下，这会增加约半秒的方向切换确认延迟，但不增加模型推理次数，也不保存帧。

- `data/calibration/default.json`：版本化档案、环境指纹、姿势稳健统计和注意力回归系数；
- `data/calibration/default.samples.json`：用于判断复杂设计是否必要的派生数值；不含帧或完整 landmarks。

校准被质量门拒绝时不会覆盖上述有效档案；失败批次的派生数值、总 RMSE 和逐点 RMSE 会写入 `data/calibration/rejected/`，用于定位问题。该目录同样被 Git 忽略，不含图像、视频或完整 landmarks。

摄像头索引、实际帧尺寸、主屏尺寸、DPI 或镜像约定变化时，旧档案会拒绝加载并要求重新校准。

每个用于比较准确率的独立批次都会在同一摄像头会话内先要求恢复到校准时的位置；`validate-live` 只有连续对齐 3 秒后才创建批次并显示第一个 trial，避免关闭预览、重开进程时的位置变化污染测试。以下独立命令仅用于提前熟悉位置要求：

```powershell
.\.venv\Scripts\python.exe -m observer align --profile data/calibration/default.json --models-dir models
```

全屏镜像预览中的十字代表当前脸部中心，绿色小框代表校准位置，黄色点用于保持视线方向一致；匹配脸部位置/大小并连续保持 3 秒后自动完成。姿势状态始终显示和记录，但默认不阻塞注意力复位；只有 `validate-live` 的姿势批次或独立命令显式增加 `--require-posture` 时才要求中立姿势也合格，避免不同能力的测试互相污染。复位步骤不保存帧，正式批次只在 `manifest.json` 记录复位耗时、最终上下文 z 和姿势状态。上下文尺度设置独立的复位下限：在默认 5z 门槛下约对应画面宽 2%、高 3% 和脸尺度 1%，避免把逐点中位数的极小回归波动误当作人重新坐下时必须复现的精度。若仍无法稳定完成，不应直接开始测试或要求用户反复“坐好”，应检查校准档案、座椅/摄像头位置和算法上下文尺度。

冻结条件下先做至少 3 个独立有效批次。若 RMSE 极差不超过 `0.03` 或变异系数不超过 `15%`，并且同一方向/点位持续占主要误差，则把批次均值视为当前程序的稳定性能：均值不达标属于系统偏差或能力下限，不再归因于个人测试动作。详细判定见 `docs/phase-c-validation-protocol.md` 9.1。

## 4. 先跑 tuning

查看内置场景：

```powershell
.\.venv\Scripts\python.exe -m observer validation-scenarios
```

为降低一次操作过长的风险，按能力分别建立 tuning 批次：

```powershell
.\.venv\Scripts\python.exe -m observer validate-live --profile data/calibration/default.json --role tuning --signals away
.\.venv\Scripts\python.exe -m observer validate-live --profile data/calibration/default.json --role tuning --signals phone_held
.\.venv\Scripts\python.exe -m observer validate-live --profile data/calibration/default.json --role tuning --signals posture_deviation
.\.venv\Scripts\python.exe -m observer validate-live --profile data/calibration/default.json --role tuning --signals attention_away
```

手机 tuning 默认只包含四个实际场景：自然工作、单手手机、双手低头手机、相似非手机物体，每项 6 秒、1 次。自然工作和非手机物体允许 compliant 或因手部不可见而 unknown，但不得聚合为 violation；两个手机场景必须为 violation。手机单能力测试不依赖注意力校准位置，因此不运行脸部位置复位门。只补测部分场景可使用：

```powershell
.\.venv\Scripts\python.exe -m observer validate-live --profile data/calibration/default.json --role tuning --signals phone_held --scenarios phone_two_hands_low_gaze,phone_non_phone_object --no-error-evidence
```

视觉场景通过后运行手机持续事件：

```powershell
.\.venv\Scripts\python.exe -m observer validate-phone-events --profile data/calibration/default.json --models-dir models --role tuning
```

完整套件包含正常工作、短暂手机、恢复、持续手机、恢复、非手机物体。若安全反例已通过，只补正例可增加 `--suite positive`。等待页和 3 秒准备页会实时显示 `phone_held`；正例应先调整到 violation 再开始。当前默认手机触发 10 秒、首次扣 5 分、可观测恢复 2 秒、不可观测安全宽限 3 秒。短暂正例要求至少连续 3 秒形成候选且不得报警；持续正例必须至少 80% violation，在 10 秒附近只报警和扣分一次；放下后可以按 2 秒 recovered 或 3 秒 observation_lost 结束。事件批次不保存图像。

窗口显示 TARGET 和 CURRENT；注意力方向场景还显示相对于本次 3 秒中立基线的 `yaw`、`pitch` 及虹膜预测 `x/y`。左、右、低头必须分别匹配对应方向并连续保持 0.6 秒，任意其他方向的 attention violation 不得冒充目标。当前水平头姿门槛为 ±0.22，虹膜单独触发左右目标时保留屏外 8% 安全余量。普通低头必须同时满足相对 `pitch >= 0.07` 和虹膜 `y > 1.0`；只有明显低头 `pitch >= 0.14` 才允许头姿单证据触发。水平方向优先级高于垂直方向：`abs(yaw) >= 0.22` 时核心逻辑会完全禁用低头证据，即使虹膜 y 或 pitch 同时异常也不得归为向下。这些都是 tuning 候选，冻结 holdout 后才能确认。看到绿色勾再按空格，经过 1 秒准备后开始记录；即使没有绿色勾也允许按空格，失败尝试不会被静默筛掉。记录期间会显示目标保持或丢失，每个场景结束后重新等待空格。绿色勾只能帮助 tuning 中理解规则，不能要求测试者扭曲自然动作去迎合，也不能作为 holdout 筛选条件。自然动作与绿色提示冲突时立即按 `Esc`：该 trial 标为 invalid 并保留失败原因，人工真值优先于自动报告；不要在同一 trial 中改做另一个动作来“完成测试”。中途 `Esc` 会取消准备或把正在执行的 trial 标为 invalid 后退出；全程不保存视频，`--no-error-evidence` 时也不保存图片。

低头参数每次修改后不能只重复真正低头正例，必须在同一冻结版本运行三个专项场景：真正看桌面、明显向任一侧看但保持视线水平、轻微低头但眼睛仍看屏幕。真正低头应持续变绿；两个对照都必须持续不变绿。2026-08-31 的实践证明，仅有低头正例 90/90 会遗漏两种可持续误判，因此 positive-only 批次不得再作为低头特异性完成证据。

历史 `attention_desk_control_chin` 要求测试者同时控制头部和眼睛，已经诱发“头部过度下压、眼睛上翻看屏幕”的阈值迎合动作；它只保留用于诊断回放，不得进入默认 tuning 或 holdout。正式安全反例改用自然注视屏幕下方目标，正式正例使用自然看桌面；测试者只按日常方式完成动作，不根据数值或颜色寻找算法边界。自然边界复测和 holdout 应关闭目标是否匹配的绿色/黄色结果反馈，只显示采集状态。若头姿和虹膜注视明显冲突，预期产品语义是注意力 `unknown`：不扣注意力分，也不计为合规；姿态继续按独立规则和持续时间处理。

姿态批次需要分别报告 `head_posture`（face-only）与 `upper_body_posture`（需要肩部/躯干）。默认场景仍包括正常、明显前倾/含胸和明显侧倾，但只有当前可观测子项参与判定：肩部不足不算违规，也不得使 face-only 结果 unknown。对齐页分行显示 FACE POSITION、FACE SIZE、FACE CONTEXT、HEAD POSTURE、UPPER BODY POSTURE 和 OVERALL；短于 0.5 秒的波动暂停进度，持续异常才清零。当前脸尺度复位实用下限为 0.005，在默认 5z 下约允许相对 0.30 基线 ±8.3%；横纵位置容差未放宽。姿态使用 7 帧、至少 5 帧的中位数以抑制关键点抖动。该分层语义在实现和复测完成前属于已确认需求，不得提前记为通过。

每个批次包含：

- `manifest.json`：环境和哈希冻结信息；
- `trials.jsonl`：人工真值、聚合预测、覆盖率、计时与 invalid 原因；
- `derived_samples.jsonl`：用于离线重算的派生数值与原始/确认手机候选，不含图像；
- `report.json`：按能力汇总的混淆矩阵和指标；
- `resources.json`：本批资源统计；
- `evidence/`：只有误判关键帧和到期索引，禁用证据时不存在。

只能依据 tuning 结果调整质量门、特征、阈值或实现。任何调整都记录原因和调整前后指标。

### 4.1 离席连续事件验证

trial 分类通过后，必须让同一个规则引擎跨场景连续运行，验证计时、报警动作、扣分和恢复，而不能用短 trial 的 F1 代替：

```powershell
.\.venv\Scripts\python.exe -m observer validate-away-events --profile data/calibration/default.json --models-dir models --role tuning
```

依次完成正常在座 8 秒、短暂离席 10 秒、返回 3 秒、持续离席 25 秒、恢复 4 秒、白纸完全遮脸 8 秒和手指完全遮镜头 5 秒。每段都在等待页按一次空格，利用 3 秒准备倒计时先形成目标动作，再在整个记录期保持。当前默认离席触发为 20 秒、恢复为 2 秒、首次扣 3 分；8 秒是注意力偏离阈值，不是离席阈值。

事件批次只保存 `event_samples.jsonl`、`event_segments.json`、`event_actions.json`、`event_report.json` 和 `resources.json` 等数值记录，不保存图像或视频。通过条件包括：正常与短暂离席不报警；持续离席只激活、报警和扣分一次，计时误差不超过 1.5 秒；返回后按配置恢复；白纸遮脸仍判人在场；手指遮镜头为 unknown；总扣分与最终分数一致。测试人员还必须核对每段 `state_counts` 是否证明动作真实形成；程序化 `passed=true` 不能掩盖动作开始过晚或保持不足。

2026-08-31 tuning 批次 `event-tuning-20260831T052607Z-429bb6a0` 的 20 秒触发、3 分扣分、约 1.97 秒恢复、遮脸在场和遮镜头 unknown 均通过；短暂离席段只形成约 4 秒可识别离席，因此只作为短时计时取消证据，不作为完整连续 10 秒证据。实际声音与 Windows 系统通知尚未接入，本命令验证的是报警动作事件。

只补测短暂离席时使用专项套件，不重复完整七段：

```powershell
.\.venv\Scripts\python.exe -m observer validate-away-events --profile data/calibration/default.json --models-dir models --role tuning --suite brief
```

专项依次记录正常基线 5 秒、短暂离席 10 秒和返回 4 秒；三段目标状态占比都必须至少 80%，不得产生激活、报警或扣分，最终分数必须为 100。批次 `event-brief-tuning-20260831T054804Z-769e90af` 实测三段目标状态均为 100%，专项通过。

## 5. 冻结并跑全新 holdout

修复 tuning 问题后重新运行自动化测试，提交代码和配置。确认将要用于 holdout 的 Git SHA、模型清单和校准档案不再变化，然后用全新动作运行：

```powershell
.\.venv\Scripts\python.exe -m observer validate-live --profile data/calibration/default.json --role holdout --signals away
.\.venv\Scripts\python.exe -m observer validate-live --profile data/calibration/default.json --role holdout --signals phone_held
.\.venv\Scripts\python.exe -m observer validate-live --profile data/calibration/default.json --role holdout --signals posture_deviation
.\.venv\Scripts\python.exe -m observer validate-live --profile data/calibration/default.json --role holdout --signals attention_away
```

不得因 holdout 失败而直接调整后继续沿用同一批。需要修改时，该批只作为失败证据，修改后重新冻结并创建新的 holdout。

## 6. 60 分钟正常工作长测

长测不主动制造违规，使用真实日常软件负载：

```powershell
.\.venv\Scripts\python.exe -m observer vision-smoke --models-dir models --profile data/calibration/default.json --seconds 3600
```

需要对自然监控中的 violation 进行人工复核时，必须显式启用有限证据并同时保存完整报告，例如：

```powershell
.\.venv\Scripts\python.exe -m observer vision-smoke --models-dir models --profile data/calibration/default.json --seconds 300 --evidence-dir data/natural/<run-id>/evidence --report data/natural/<run-id>/report.json
```

只要传入 `--profile`，正式计时前必须先显示 3 秒 `SESSION SETUP - NO RECORDING`：测试者保持自然工作距离、完整脸和双眼可见并看屏幕中心。该步骤只建立本次摄像头会话的中立 yaw/pitch 基线，不要求回到历史校准脸尺度、不要求肩膀可见，也不计入监控时长或调试录像。报告的 `session_attention_setup` 必须包含两项基线和有效观测数；逐帧 attention 指标必须包含 `attention_head_yaw_delta` 与 `attention_head_pitch_delta`。缺少这些字段的自然注意力批次按退化运行处理，不得进入验收指标。

该模式不录视频、不保存合规帧，只为离席、手机、注意力、姿态和人体工学 violation 各保存最多 3 张常规 JPEG，同一信号相邻常规截图至少 30 秒；如果连续 violation 达到该信号的产品触发时间，再额外保留 1 张 `sustained_threshold` 证据。`report.json` 的 `evidence.records` 给出每张图的绝对路径、发生时刻、取证类型、派生指标、SHA-256 和到期时间。先由测试者结合当时真实动作审核图像，再决定是否调整阈值；没有图像的历史统计不能单独作为调参依据。

当有限关键帧无法分析“算法没有检出的动作”时，可由项目所有者逐轮明确授权 `--debug-video <new-path.mp4>`。该选项临时保存本轮完整本地 MP4、逐帧真实时间索引和 JSONL 信号/派生量轨迹，包括原始与跟踪后的手机候选；默认关闭，不得沿用为日常监控配置。文件标记为 review-then-delete，建议 24 小时内完成复核；分析完成后必须列出精确文件并经项目所有者确认再删除。调试录像的 CPU、磁盘和文件大小单独记录，不得混入默认无录像资源承诺。

需要精确动作时长时同时使用 `--guided-debug-cues`。窗口只显示明显的 `STEP n/7`、短动作说明、剩余秒数和原始摄像头取景，不显示 compliant/violation/unknown 或任何阈值反馈；阶段切换发出一次系统提示音。非手机阶段必须明确显示 `NO PHONE`，避免动作跨窗口污染；逐帧 trace 同时记录 `cue_id` 和剩余时间，供录像与算法输出客观对齐。固定 300 秒流程为：0～180 秒自然专注，180～205 秒近距离看屏幕下缘，205～220 秒恢复，220～232 秒明确看出屏幕左侧，232～245 秒恢复，245～260 秒手机清晰入镜自然使用，260～300 秒恢复自然专注。测试者只执行当前窗口动作；如果动作提前、延后或跨越其他窗口，相应专项必须标为无效，不得用后段正确动作覆盖前段错误输入。不得用主观计数替代该时间真值，也不得根据模型结果调整动作。

注意力会话基线修复使用一分钟专项 `--guided-debug-plan attention-minute`，不运行五分钟流程：0～10 秒正常看屏幕中心；10～25 秒明确看向屏幕外左下方；25～35 秒恢复；35～50 秒自然靠近但只看屏幕内左下区域；50～60 秒恢复。第一段是持续离屏正例，第二段是近距离屏内反例；两项必须在同一轮同时成立，不能用简单降低阈值换取正例召回。该一分钟之前的 3 秒会话设置不计入 60 秒。

早期定时 pilot 使用 `--guided-debug-plan attention-boundary`，计划总长固定为 151 秒。它证明了连续脚本和逐用例报告链路可工作，但固定 3 秒准备可能早于测试者真正就绪，所以该现有计划只保留为 pilot/录像诊断，不能直接作为正式连续 holdout。

下一轮 P0-A 先建立交互基线。启动前逐项向测试者说明动作、参照点、距离、期望、计分时长和通过标准；标准有疑问时先不测试。`validate-live` 显示简短 `STANDARD`、黄色 `TARGET NOT READY` 或绿色 `TARGET READY`；测试者以真实动作为准，确认后按空格或回车，经过 1 秒稳定期才记录。ESC 表示标准/动作有疑问并取消批次。等待和稳定期不计分、不进入 trial，也不写入调试小视频。近距离需要脸尺度至少比校准尺度大约 `0.10`；远距离目标为校准尺度上下 `0.03`。

开测对齐阶段的绿色人脸位置框水平居中于 `0.500`，容差为画面宽度的 ±5%。历史档案中的脸部水平位置只表示当时坐在哪里，不能迫使后续测试者复现一个偏移且仅数像素宽的点；启动门同样不要求自然工作距离回到历史脸尺度，而是保存本次会话自然脸尺度，供后续“再靠近”用例作相对参照。校准远距离用例仍明确回到历史尺度。完整脸、双眼和垂直取景继续检查。该门禁修复不更改 17 点注视模型。

```powershell
.\.venv\Scripts\python.exe -m observer validate-live --models-dir models --profile data/calibration/default.json --role tuning --signals attention_away --scenarios attention_boundary_a01_center,attention_boundary_a02_outside_left,attention_boundary_a03_recovery,attention_boundary_a04_outside_right,attention_boundary_a05_recovery,attention_boundary_a06_near_left,attention_boundary_a07_near_right,attention_boundary_a08_far_center,attention_boundary_a09_far_left,attention_boundary_a10_far_right,attention_boundary_a11_final --output-root data/formal/attention-boundary-interactive --debug-trial-videos
```

该交互批次用于 tuning、确认标准和建立边界基线，不因绿色筛选而宣称 holdout 通过。正确自然的动作始终不变绿时仍可按实际真值采样，结果记为边界/漏检候选，不得迎合。每个错误 trial 保存证据帧，全部计分段保存派生数值；显式的 `--debug-trial-videos` 只录制确认并完成 1 秒稳定准备后的正式计分段，生成的本地小视频仍按 review-then-delete 处理。

当需要复现“头向一侧但眼睛补偿看回中心”以及近距离左右点混淆时，使用以下 P0-A-D 专项。绿色徽标为 `CAPTURE READY`，只表示画面/脸/眼睛/距离可采样；模型意见单列为只读诊断，不控制绿色。按键后准备和计分阶段隐藏模型结果。

```powershell
.\.venv\Scripts\python.exe -m observer validate-live --models-dir models --profile data/calibration/default.json --role tuning --signals attention_away --scenarios attention_compensation_d01_center,attention_compensation_d02_head_left_eyes_center,attention_compensation_d03_outside_left,attention_compensation_d04_recovery,attention_compensation_d05_head_right_eyes_center,attention_compensation_d06_outside_right,attention_compensation_d07_recovery,attention_compensation_d08_near_center,attention_compensation_d09_near_left,attention_compensation_d10_near_right,attention_compensation_d11_final --output-root data/formal/attention-compensation-interactive --debug-trial-videos
```

执行前固定五个视线点：屏幕中心；从显示区域边缘向内约 2～3 厘米的左下、右下点；位于左右边框外约 10 厘米、并与显示区域下方四分之一高度对齐的左下、右下点。循环 1 和循环 2 不移动参照点、摄像头、电脑或座椅，并分别重启程序以获得独立会话基线。

距离分为三档：本轮会话设置时的自然舒适距离；比自然距离明显更近但完整脸和双眼仍在画面中的近距离；以及当前校准档案对应的远距离，目标脸尺度约为 `0.270`，不要求退到眼睛无法检测。当前 17 点及 8 个独立验证点无需重做，它们仍是同一摄像头/屏幕环境下的坐标锚点；P0-A 只用中心和左右下边界代表点检查距离泛化。若远距离或近距离代表点失败，才建立该距离的专项网格，不预先重复三套 17 点校准。

交互基线通过、标准争议关闭并冻结算法后，才生成正式连续脚本。整组启动前一次性确认动作顺序和停止条件，运行中不显示黄/绿、`CURRENT` 或实时得分，不要求再次按键；程序独立跑完后统一分析。动作切换窗口预先写入脚本，转换帧不计入稳定段。两个 5 分钟自然循环也采用同样的“开始前确认一次、中间不干扰”原则。

以下命令仅是旧 pilot 的复现命令，不得原样用于新的正式连续基线：

```powershell
.\.venv\Scripts\python.exe -m observer vision-smoke --models-dir models --profile data/calibration/default.json --guided-debug-plan attention-boundary --debug-video data/formal/attention-boundary-cycle-1/session.mp4 --evidence-dir data/formal/attention-boundary-cycle-1/evidence --report data/formal/attention-boundary-cycle-1/report.json
```

循环 2 必须在循环 1 完全退出后使用新目录启动：

```powershell
.\.venv\Scripts\python.exe -m observer vision-smoke --models-dir models --profile data/calibration/default.json --guided-debug-plan attention-boundary --debug-video data/formal/attention-boundary-cycle-2/session.mp4 --evidence-dir data/formal/attention-boundary-cycle-2/evidence --report data/formal/attention-boundary-cycle-2/report.json
```

两个旧命令不会自动连续执行。正式连续脚本生成后也必须采用两个独立摄像头会话，轮间只确认设备释放和输入有效，不得修改参数；每一轮内部不中断、不确认。

姿态判定不得把同一次距离变化复制成多个独立证据。绝对脸部位置、脸尺度和肩宽不参与违规投票；`head_offset_y` 与 `face_to_shoulder_scale` 也不得自行组合成违规。有效组合必须包含真正的角度、肩线或躯干佐证：前倾为头部俯仰加垂直/比例证据，侧倾为歪头加肩线/垂直证据，上身侧倾为肩线加垂直证据；face-only 时要求俯仰与歪头同时明显异常。报告中的 `reason` 应明确给出采用的证据组合。

记录输出中的 CPU 平均/P95、内存峰值/增长、各模型延迟、摄像头错误和最终信号。正式误报率以真实验证批次和人工复核为准，不能只看最后一帧。

## 7. 复杂度复核与证据清理

使用同一批 `default.samples.json` 和 `derived_samples.jsonl` 完成 `docs/phase-c-validation-handoff.md` 中 CX-01～CX-08 的 N/C 评分。先做离线关闭或简化对照，不改变 holdout 标签。结论为 remove/defer/simplify 的项目必须按清理门处理后，才生成最终 `docs/phase-c-result.md`。

查看到期证据而不删除：

```powershell
.\.venv\Scripts\python.exe -m observer evidence-purge data/validation/<batch-id>/evidence/index.jsonl
```

确认列表后显式删除：

```powershell
.\.venv\Scripts\python.exe -m observer evidence-purge data/validation/<batch-id>/evidence/index.jsonl --apply
```

阶段 C 结果只能是通过、有条件通过或返工；真实指标缺失时保持“实施中”，不能用自动化测试数量代替。
