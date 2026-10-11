# F 内部候选使用及构建说明

日期：2026-10-10（America/Denver）

## 启动与数据

保留候选的整个 `Observer` 文件夹，双击 `Observer.exe` 打开 GUI。实际候选编号及验证状态以 [F 结果报告](phase-f-result-20261010.md) 为准。真实摄像头补验继续暂缓；模拟点验可在 PowerShell 中运行：

```powershell
& 'C:\候选路径\Observer\Observer.exe' product --data-dir 'C:\你的隔离目录' gui --demo
```

诊断控制台为同目录的 `Observer-diagnostic.exe`，支持相同产品命令。默认资源位于 `_internal/models` 与 `_internal/observer/default_config.json`，默认产品数据为 `%LOCALAPPDATA%/Observer/product`，个人校准为 `%LOCALAPPDATA%/Observer/calibration/default.json`。显式路径优先；不自动搬移源码版数据。候选没有附带个人校准。

运行中的关闭按钮在托盘可用时隐藏窗口；“退出程序”释放工作线程和运行锁后退出。只有完全退出后才能替换整个程序目录；用户数据目录应独立保留。当前同版本候选目录替换的证据不承诺未来 schema 降级兼容，也不授权真实用户数据迁移。

窗口版启动错误写到数据目录的 `logs/launcher.log`（512 KiB，保留一个轮转文件）；工作线程准备结果为 `logs/startup.jsonl`。无法创建日志时，启动错误对话框显示原因。对日志的判断应结合界面实际状态；通知“提交”及声音字节生成均不能替代人工送达/可听确认。

## 重新构建

前提：Windows x64、项目 Python 3.11、uv、完整源码、`uv.lock` 及已经校验的四个冻结模型。当前构建环境使用 Python 3.11.15。从仓库根目录执行：

```powershell
.\scripts\build_windows.ps1 -BuildId '自选唯一编号'
```

脚本按 `uv.lock` 准备忽略目录 `build/phase-f-env` 的 desktop/yolo 运行依赖，再按 `packaging/requirements.lock` 的版本和哈希安装构建工具。原 `.venv` 不作为打包内容。打包的依赖扫描 PATH 只包含构建解释器、Windows System32 及 Windows 目录，避免收集其他应用的同名 DLL。

产物为 `dist/<编号>/Observer`；构建日志、命令、输入身份和文件清单位于 `artifacts/phase-f/<编号>`。构建要求编号对应的目录不存在；不要通过覆盖旧目录掩盖旧结果。构建期间不得更改打包输入；脚本发现输入摘要变动会拒绝候选。

默认配置与模型原字节带入包内。Windows 非 ASCII 模型路径使用 MediaPipe 官方 `model_asset_buffer` 加载，ASCII 路径保留原文件加载方式；配置、模型身份及推理阈值不改变。

## 自动候选验证

```powershell
.\.venv\Scripts\python.exe scripts/test_windows_candidate.py dist/<编号>/Observer --run-id <唯一测试编号>
.\.venv\Scripts\python.exe scripts/test_windows_candidate.py dist/<编号>/Observer --run-id <另一唯一编号> --native-qt
.\scripts\test_candidate_readonly.ps1 -Candidate dist/<编号>/Observer -RunId <唯一只读测试编号>
```

这些验证建立合成数据、合成小图及候选副本，不打开摄像头。验证进程去除开发环境 PATH/PYTHONPATH；Qt 单实例监听需要正常登录用户的本地管道权限。沙箱拒绝管道时，记录权限错误并在已授权环境比较，不能删减 IPC 测试取得通过。

只读测试仅修改独立候选副本的 ACL，先验证写入确实被拒绝，随后运行候选，最后恢复原 ACL 并记录恢复结果。测试退出码非零时检查证据，不能继续宣称通过。合成截图及事件图片只留在 `build/phase-f-tests` 的隔离数据中，不进入候选分发目录。

各项实际结果、不可替代的人工确认、干净机器验证及 E 遗留门见结果报告。安装器、正式发布和第三方许可签收仍需后续阶段工作。

## 独立 Windows 验收入口

独立验收包把原候选完整收进 `Observer/`，另附 `validate_windows_portable.ps1` 和 [人工点验说明](../packaging/PORTABLE-VALIDATION.txt)。工具位于候选目录外，不更改 f005 的冻结输入身份或文件清单。转移包仅供已授权内部验证；许可签收仍待办。

从仓库导出可用：

```powershell
.\.venv\Scripts\python.exe scripts/export_windows_validation.py dist/<编号>/Observer --export-id <唯一导出编号>
```

导出前核对候选清单；完成后复核 ZIP CRC 及压缩包内每项 SHA-256。ZIP 位于 `dist/<导出编号>.zip`，转移摘要位于 `artifacts/phase-f/<导出编号>/transfer-receipt.json`。脚本拒绝覆盖既有输出。

接收端解压到中文/空格路径，在解压目录用 64 位 Windows PowerShell 5.1 或 PowerShell 7 执行：

```powershell
.\validate_windows_portable.ps1 -IndependentMachine
```

接收端不需要仓库、Python、uv 或管理员权限。测试对子进程清理开发 PATH/环境变量，从无关目录启动，使用新的隔离 LocalAppData 和临时目录，进行模型合成推理及 Windows 原生 GUI 自检，最后确认候选完整性。父进程环境、执行策略和真实用户数据不变。系统策略拒绝脚本时保留错误，按该设备管理规则处理，不要求降低持久策略。

结果写入 `validation-results/<唯一编号>`。`-IndependentMachine` 只记录操作者声明，不能自动证明机器缺少开发环境；需要独立机器实际运行记录及人工观察才能签收 F-P2。`manual-results.json` 起始为待测，请保留自动报告原始内容。人工首次启动使用显式 `--demo` 的测试快捷方式；真实无参数冷启动涉及摄像头，继续保留到 E 补验恢复后。
