# F Windows 内部候选结果

日期：2026-10-10（America/Denver）
状态：**F 实施中；内部 onedir 候选及首轮自动验证已完成。阶段签收、安装器和正式发布尚未完成。**

E 继续“实施中（补验暂缓）”。设备中断/恢复补验及修复后 60 分钟长测保留待测；本轮没有打开摄像头、切换设备、重启电脑或建立续跑任务。

最新提示音调整及 f006 候选见第 6 节；第 1～5 节保留 f005 首轮基线证据。

## 1. 首轮基线交付（f005）

首轮基线候选：`C:/Qproject/Observer/dist/f005-20261010/Observer`。

- 无控制台入口 `Observer.exe`；诊断入口 `Observer-diagnostic.exe`；共用 `_internal`，无双份模型。
- [最短启动说明](../dist/f005-20261010/Observer/START-HERE.txt)、[详细运行/构建说明](phase-f-runbook.md)。
- [spec](../packaging/Observer.spec)、[构建脚本](../scripts/build_windows.ps1)、[工具锁](../packaging/requirements.lock)。
- [构建来源记录](../artifacts/phase-f/f005-20261010/build-info.json)、[实际构建命令](../artifacts/phase-f/f005-20261010/command.json)、[依赖扫描环境](../artifacts/phase-f/f005-20261010/build-environment.json)、[构建日志](../artifacts/phase-f/f005-20261010/build.log)。
- [文件及 SHA-256 清单](../artifacts/phase-f/f005-20261010/files.sha256.json)、[最后完整性复核](../artifacts/phase-f/f005-final-integrity.json)。清单不包含清单自身；记录 1,177 个文件、488,650,968 字节（约 466 MiB），另有清单自身 206,730 字节。
- [第三方许可范围和未审查项](../packaging/REDISTRIBUTION.md)；候选内 `_internal/licenses/inventory.json` 及可发现的许可文件。清单包含构建环境依赖，不等于全部运行组件的许可签收或完整 SBOM。

构建环境为 Windows x64 / Python 3.11.15 / PyInstaller 6.22.3 / hooks-contrib 2026.8。运行环境沿用锁文件：PySide6 6.11.2、MediaPipe 0.10.21、NumPy 1.26.4、OpenCV contrib 4.11.0.86、ONNX Runtime 1.29.0、psutil 7.2.2。没有升级现有运行依赖，工具单独安装在 `build/phase-f-env`。

Git 基底为 `6b37f255318b2824fe4294a7d8378e35fe22a51e`，工作区有未提交改动；不声称候选等于该提交。构建输入的每个源码文件、资源及脚本哈希均记录，源码树摘要为 `49aa3165762217379f0146e95f088aab9cf04df5c69f3dad85c05f1c76c32ed2`。会话保存 `source_identity_origin=build-input-manifest` 和构建记录摘要，未使用冻结版空源码目录摘要。

## 2. 实现及已定位问题

冻结入口无参数进入 GUI；显式产品 CLI 参数仍可用于诊断。模型及默认配置从包内读取，默认产品目录为 `%LOCALAPPDATA%/Observer/product`，默认校准为 `%LOCALAPPDATA%/Observer/calibration/default.json`。显式路径优先；源码默认路径兼容现状，没有搬移真实用户数据。窗口版通过有界本地启动日志及启动错误对话框报告错误。

首批候选在 QtWidgets 导入时报告 Windows 错误 127。PE 导入/导出和构建 TOC 查明：依赖扫描拾取了 Codex 工具环境 Poppler 的 ICU 78，导出 `ucnv_open_78` 等符号，Qt 实际需要 `ucnv_open` 等不带后缀符号。单纯添加 DLL 搜索路径没有解决；候选副本中临时移开该 DLL 后，Qt 能导入，随后的失败变为已知沙箱 IPC 拒绝访问，见 [对照](../artifacts/phase-f/f003-icu-probe.json) 和 [修正了 PE 导出枚举上限的符号报告](../artifacts/phase-f/f003-dll-imports-corrected.json)。最终构建进程清理自身 PATH，避免收集其他工具的同名 DLL；没有改变用户或系统的持久配置。f001～f004 及失败证据保留，不作为可交付候选。

随后中文目录的真实模型加载暴露 MediaPipe 原生路径读取问题。[最小对照](../artifacts/phase-f/mediapipe-unicode-probe.json)中，同一 Face 模型英文路径成功、中文路径失败、原字节缓冲成功。接线仅对 Windows 非 ASCII 路径使用官方 [BaseOptions.model_asset_buffer](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/BaseOptions)，英文路径保留原方式。Face、Pose 和 EfficientDet 均沿用原字节；冻结手机模型、配置哈希、推理参数、评分和 schema 未变。最终候选在中文路径实际加载四个模型并处理合成帧通过；该证据不涉及真实摄像头或性能长测。

初始文件审计误报了第三方 OpenCV `data` 模块及 Matplotlib 示例图片；检查其第三方来源后收窄了路径规则，未删除测试或用户证据。沙箱内 IPC 监听和默认 Temp 写入也出现拒绝访问，日志保留；正常权限下使用项目内临时目录复核，没有放宽沙箱配置。

## 3. 验收结果

自动测试命令均在日志或 JSON 记录实际执行入口、时间及退出码。所有 UI/资源测试使用合成数据及随机/零值合成图片，未进行摄像头采集；图片不进入发行目录。

| ID | 实际结果 | 证据及限制 |
| --- | --- | --- |
| F-P1 | 通过 | [完整回归](../artifacts/phase-f/unittest-final-20261010.log)：348 项，88.132 秒，退出 0，无失败、错误或跳过。较交接的 339 项增加 7 项部署接线及 2 项模型路径用例；原测试完整保留。冻结资源通过现有校验。 |
| F-P2 | 本机自动部分通过；干净机器待测 | 新独立环境按 `uv.lock` 安装，工具哈希锁单独安装，PyInstaller `--clean` 构建退出 0。执行时 PATH 仅 System32，去掉 PYTHONPATH/PYTHONHOME/Qt 开发路径；候选副本从任意目录运行通过。没有在缺少仓库、Python 和开发环境的另一台 Windows 机器上验证，不能写“干净机器通过”。 |
| F-P3 | 自动部分通过；首次双击人工点验待测 | [离屏测试](../artifacts/phase-f/f005-approved/summary.json)、[Windows 原生测试](../artifacts/phase-f/f005-native-approved/summary.json)：中文/空格程序及数据路径、任意工作目录、包内模型实际加载和 GUI 闭环通过。无参数第二进程成功唤醒既有所有者。冷启动的真实双击及快捷方式起始目录尚未人工点验。 |
| F-P4 | 通过 | 诊断/GUI EXE 的开始→暂停→继续→结束及历史重读均通过；每轮手动会话只生成一条已结束记录，随后短定时任务另生成一条。退出后运行锁可重新取得。 |
| F-P5 | 自动部分通过；托盘人工操作待测 | 第二实际 EXE 唤醒已有窗口，无参数入口也唤醒，未新增第二会话；明确退出后释放运行锁，再打开历史保持。原生 Qt 界面正常。真实托盘图标可见性、关闭隐藏/托盘菜单恢复和退出仍需人工点验。 |
| F-P6 | 自动部分通过；送达和可听确认待测 | 每次定时任务只有一条关联会话和一次通知调用；通知开关跨窗口重开保持；两种提醒波形可生成。Windows 通知实际展示/点击恢复和声音实际可听尚未人工确认。 |
| F-P7 | 已执行自动项通过 | [故障注入](../artifacts/phase-f/f005-faults/summary.json)：配置/Face 模型改一字节、缺失 Qt 模块均返回 1 并有诊断，未创建产品数据库，副本资源恢复。主测试中的缺手机模型也返回 1。 [只读副本](../artifacts/phase-f/f005-readonly-approved/summary.json)：写入探针确实被拒绝，模型推理及模拟闭环退出 0，原 ACL 恢复。 [无控制台句柄](../artifacts/phase-f/f005-no-console-approved/summary.json)：普通用户 token、NULL stdio、不继承句柄，GUI 闭环退出 0，结果写入 launcher.log。GUI 致命错误对话框的实际显示仍可随冷启动人工点验。 |
| F-P8 | 同版本目录替换/返回通过 | 合成数据依次由中文副本、原候选目录、中文副本运行，历史从 2→4→6 条，既有 ID、设置和合成证据哈希保留，程序目录文件摘要未变化；所有操作在测试所有者退出后执行。未测试跨 schema 降级或真实用户迁移。 |
| F-P9 | 文件/隐私审计通过；许可签收待办 | 清单无用户数据库、个人校准、用户照片、训练集、日志、.env、.git、.venv 或 E 注入脚本。默认配置/四模型哈希一致；最终候选 1,177 项摘要无变化、无额外或缺失文件。模型及第三方分发条件另列未审查，不能公开分发。 |
| F-P10 | 保留待测 | 沿用 E 暂缓决定，未做真实摄像头及资源长测；合成帧不替代新鲜摄像头帧、释放或 60 分钟门。 |

离屏 Qt 渲染不代表原生字体效果；原生 `qt_platform=windows` 的[合成界面截图](../artifacts/phase-f/f005-native-approved/synthetic-desktop.png)已实际查看，中文、布局及按钮正常。截图是内部模拟结果，不属于真人验证。

## 4. 后续必要项

1. 在冻结候选上人工确认首次双击、关闭隐藏、托盘恢复/退出、Windows 通知展示及声音可听；使用隔离模拟目录，按 [运行说明](phase-f-runbook.md) 操作。
2. 在没有项目 Python/仓库/开发 PATH 的独立 Windows 环境验证，记录系统及运行结果。
3. 待项目所有者明确恢复 E 补验后，先确认摄像头可采集新鲜帧，再补设备恢复及受控 60 分钟长测；打包版真实模式也需要 F-P10 证据。
4. 审查 Python、Qt/PySide、MediaPipe/模型、YOLO 来源及各原生依赖的分发条件。公开发布仍需 E 遗留门、打包验收及项目所有者签收。

首轮停在可审查的便携内部候选。安装器尚未实施，自动启动/更新及其他 F 功能未扩展。旧 E 改动、原始失败日志和无关用户文件保留；本轮没有 Git 提交或推送。

## 5. 独立验收转移准备（继续工作）

项目所有者确认暂时没有独立 Windows 环境，**独立机器验收继续待测**。为后续验收新增候选目录外的 [PowerShell 验收入口](../scripts/validate_windows_portable.ps1)、[导出脚本](../scripts/export_windows_validation.py) 和 [独立环境/人工点验说明](../packaging/PORTABLE-VALIDATION.txt)。这些是外部验证工具，未修改 f005 的冻结 EXE、资源、输入身份或候选清单；它们也不随原候选入口运行。

[内部转移 ZIP](../dist/f005-portable-validation-20261010.zip) 为 218,757,566 字节（约 209 MiB），包含原 `Observer/` 及两份验证工具/说明。导出前核对原候选 1,177 项摘要，导出后校验 ZIP CRC、包内每项 SHA-256、原始文件清单及工具摘要。完整摘要见 [转移记录](../artifacts/phase-f/f005-portable-validation-20261010/transfer-receipt.json)，ZIP SHA-256 为 `1fec8fab17ead8a6f0143d8c2d2042cf29d964578eef04aacda05fcf024f2155`。

验收入口在本机 Windows PowerShell 5.1.26100.9444、64 位普通用户 token 下实际通过，[本机入口证据](../artifacts/phase-f/portable-runner-approved/20261011T015718626Z-b1ff701b/summary.json)。它不使用 Python/uv/仓库脚本，对 EXE 子进程清理开发环境，执行四模型合成推理及 Windows 原生 GUI 自检、无参数第二实例唤醒和历史重开，前后验证候选完整性。首次沙箱调用报 `AuthorizationManager check failed`；最小比较中的 Windows PowerShell 执行策略读取报注册表访问不允许，正常权限的同一 `-File` 调用通过。没有修改执行策略、注册表、用户/系统 PATH，也没有声称沙箱已修复。

转移 ZIP 另解压到项目内新的中文/空格路径，直接运行包内脚本的默认相邻 `Observer/` 及默认输出路径，两个 EXE 返回 0，1,177 项文件前后校验一致，普通用户 token，见[解压往返证据](../artifacts/phase-f/portable-export-roundtrip-20261010/summary.json)。该次调用只依赖解压包及 Windows PowerShell；仍是本机环境验证，不是独立机器验收。

`-IndependentMachine` 参数只记录操作者声明；脚本明确将 `independent_machine_verified_automatically` 保持为 false，并将人工点验模板初始设为 pending。本机结果不替代 F-P2，通知提交和声音生成不替代实际展示/可听确认。无参数真实冷启动涉及摄像头，继续暂缓；人工冷启动先用显式 `--demo` 的测试快捷方式。

## 6. 提示音调整（项目所有者要求）

2026-10-10，项目所有者要求违规两声、动作不正一声。桌面接线沿用现有分类：手机/离席等违规播放两声 880 Hz 短音，每声 120 ms，中间静音 80 ms，整体 320 ms；姿态提醒播放一声 660 Hz、120 ms 柔和短音，原音量不变。Windows 内存播放把两声放进同一 WAV；备用 Qt 提示在违规时延迟补第二声。没有修改规则、评分、事件、模型、schema 或系统任务通知。

新增 Windows 播放输入的音段数量验证，覆盖手机、离席及姿态；补充播放失败时备用一声/两声验证。首次沙箱桌面回归 77 项中，9 项 IPC 监听受权限限制失败，声音用例均通过；正常权限下[完整 349 项回归](../artifacts/phase-f/sound-change-full-approved-20261010.log)通过，86.170 秒，退出 0。

预听文件为[违规两声](../artifacts/phase-f/sound-preview-20261010/violation-two-beeps.wav)、[姿态一声](../artifacts/phase-f/sound-preview-20261010/posture-one-beep.wav)。自动检查证明波形及播放调用符合一声/两声要求；本机实际声音设备的可听确认仍由项目所有者点验。

最新候选为 `C:/Qproject/Observer/dist/f006-sound-20261010/Observer`，打包退出 0，1,177 项文件共 488,651,354 字节；[构建记录](../artifacts/phase-f/f006-sound-20261010/build-info.json)的 50 项源码/输入摘要复核一致，源码树摘要为 `e81b843f00072959fc17ee4f9f40dd99c6b42e0cabf360949a0fe6945c368799`。f005 及原证据保留，不覆盖。

[新版内部 ZIP](../dist/f006-sound-validation-20261010.zip) 已导出，218,760,036 字节，SHA-256 为 `8370f55f3c95c0b3d21fc5ac873d47e13d992da250b6f5226257454aba10ab47`，ZIP CRC 和包内全部文件摘要通过，见[转移记录](../artifacts/phase-f/f006-sound-validation-20261010/transfer-receipt.json)。该 ZIP 在本机解压到中文/空格路径，包内 PowerShell 5.1 验收入口实际运行通过：[原生包内验证](../artifacts/phase-f/f006-sound-native-20261010/20261011T021739495Z-da50ef2d/summary.json)。四模型合成推理、诊断/窗口 EXE 的模拟闭环、单实例唤醒、历史和合成证据保留、前后 1,177 项完整性校验通过；普通用户 token，不开摄像头。

当前可采用 f006 进行内部验证。f005 的故障注入/只读/空标准句柄证据继续作为首轮基线，未声称在 f006 上重复这些测试。独立机器、人工送达/可听确认、许可签收及 E 两项遗留门仍按原安排待测，尚未正式发布。

## 7. 私有 GitHub 预发布授权

2026-10-10，项目所有者要求先打包发布一个版本并同步 GitHub。现有目标 `Wq5881898/observer` 已确认是私有仓库，没有既有 Release。按新授权准备 `v0.1.0-preview.1`，保留已验证 f006 ZIP 的原始字节，附校验文件和构建记录，并同步对应源码与必要测试/文档。详见[版本发布说明](releases/v0.1.0-preview.1.md)。

此次为私有预发布；不把 E/F 未完成项改记通过，不更改仓库可见性，不启动摄像头或长测。旧包和验证证据保持可追溯，个人数据及无关文件不进入本次新增同步范围。
