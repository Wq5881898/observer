# D 极简首版使用说明

从项目根目录运行。提供极简桌面窗口和原控制台；报告导出、重复任务和复杂配置均未实现。

## 图形窗口（推荐）

已安装 PySide6；在新环境中使用 `uv sync --extra desktop --extra yolo` 安装锁定依赖。

```powershell
# 模拟体验：不打开摄像头，使用独立模拟记录
.\.venv\Scripts\python.exe -m observer product --data-dir data/product-demo gui --demo

# 真实观察：点击开始或任务到点后打开摄像头
.\.venv\Scripts\python.exe -m observer product gui
```

填写名称和时长后点“开始”。勾选“指定开始时间”后，按钮变成“安排任务”，到点自动开始；保持窗口打开。暂停/继续和结束直接点按钮。历史记录点击查看摘要；计划页可取消任务，冲突、失败或错过的任务可在没有活动会话时点“重试选中任务”立即重新执行。截图默认关闭，可在窗口勾选开启。

模拟模式另有模拟信号选择框，开始后可切换“可见手机”并等待10秒测试提醒。关闭窗口会结束当前会话并标记中断、释放摄像头；不驻留托盘。界面与推理/SQLite分线程，初始化模型期间仍可操作关闭窗口，窗口会等待资源释放完成再退出。

若存在 `data/calibration/default.json`，真实模式默认自动采用；也可用 `--profile` 指定其他档案，或用 `--no-profile` 明确关闭软信号。界面会显示当前能力，软提醒触发时播放Windows警示音、闪烁任务栏，并保留红色提醒横幅直到下一次开始；会话摘要显示非零软信号累计异常时间，累计值不等于规则要求的连续触发时间。单次任务启动失败显示原因，不会静默换模型。

启用校准时，任务状态下方同时显示即时“头眼：屏内/屏外候选/不可判断”。“屏外候选”表示当前观测为violation，必须连续达到8秒才成为提醒事件；“不可判断”不会累计事件时间。产品会话记录实际同会话头姿基线，便于与C受控验证对照。

桌面基础版属于经用户要求提前推进的 E 界面工作；不表示完整 E（完整系统通知、托盘、摄像头进程隔离等）已完成。真实图形窗口的手动开始/关闭中断、单次定时和带profile注意力对照均已通过，完整负载验收仍待进行。

## 1. 先体验（不使用摄像头）

```powershell
.\.venv\Scripts\python.exe -m observer product --data-dir data/product-demo run --demo
```

运行窗口输入：

```text
start 2 阅读
signal phone_held
```

保持约10秒后应看到一次手机提醒和扣分。输入 `signal normal` 模拟恢复；`pause` 暂停、`resume` 继续、`status` 查看当前摘要、`end` 结束并显示结果、`history` 查看历史、`quit` 退出。

模拟记录有明确标记，不能当真人准确率证据。建议使用单独 product-demo 目录。输入可用引号包含空格标题。信号命令只在模拟模式有效。

## 2. 真实运行

```powershell
.\.venv\Scripts\python.exe -m observer product run
```

再输入 `start 25 阅读`。未开始且没有到期任务时不打开摄像头；暂停和结束后关闭摄像头。暂停不顺延计划结束时间。控制台一直保留，才能执行到期任务。

真实入口检查 `models/yolo/observer-phone-s304-e2b.onnx` 的冻结SHA-256，使用YOLO手机和低频EfficientDet Person，不静默切换模型。模型缺失/哈希不符时开始失败，不产生扣分。默认校准档案存在时自动启用attention/posture/ergonomic软信号；缺失或显式使用 `--no-profile` 时，界面明确说明仅启用硬信号。硬在席路径仍保留已有无校准姿态可信门。

指定其他已有校准时：

```powershell
.\.venv\Scripts\python.exe -m observer product run --profile data/calibration/default.json
```

路径必须指向本机有效档案；每次打开摄像头会沿用 C 的临时会话基线窗口，不重新做17点校准。档案环境不匹配则开始失败。软信号始终零扣分。

明确只使用离席和可见手机时：

```powershell
.\.venv\Scripts\python.exe -m observer product gui --no-profile
```

## 3. 一次任务

保持真实运行窗口开启，在另一个终端添加任务（替换为未来实际时间）：

```powershell
.\.venv\Scripts\python.exe -m observer product task "阅读" --at "2026-09-20T09:00:00-06:00" --minutes 25
.\.venv\Scripts\python.exe -m observer product tasks
.\.venv\Scripts\python.exe -m observer product cancel <任务ID>
.\.venv\Scripts\python.exe -m observer product retry <任务ID>
```

时间必须包含UTC偏移。运行中到期60秒内启动一次；程序重启时迟到不超过60秒的pending任务仍补启动，超过窗口记missed。已存在会话则记conflict，模型/摄像头启动失败记failed，并保存简短原因。conflict、missed和failed可重试或取消；已经建立过会话的任务不会重复执行。当前不提供编辑，取消后新建即可。暂停中的会话同样占用唯一会话位置。

不同数据目录是不同任务列表；模拟任务请在命令中使用相同 `--data-dir data/product-demo`。

## 4. 历史与隐私

```powershell
.\.venv\Scripts\python.exe -m observer product history
.\.venv\Scripts\python.exe -m observer product show <会话ID>
.\.venv\Scripts\python.exe -m observer product images on
.\.venv\Scripts\python.exe -m observer product images off
.\.venv\Scripts\python.exe -m observer product delete <会话ID>
.\.venv\Scripts\python.exe -m observer product purge
.\.venv\Scripts\python.exe -m observer product delete-all
```

摘要显示状态、总分或证据不足、覆盖率、主要事件与各项时长。分数只反映离席/可见手机硬规则；硬规则有效覆盖低于80%不展示有效总分，实际账本仍可查。没有观测不能得到有效100分。短暂状态切换区间保守记为unknown。

截图默认关闭；开启后只对硬事件成立保存一张JPEG，不记录连续视频。每次运行启动及每小时按30天清理已结束会话；也可手动purge。配额满、没有新鲜帧或编码失败记录原因，不撤销事件。单次删除同时处理图片及临时文件，失败保留索引供重试。delete-all要求先退出运行窗口，删除所有产品会话、图像及任务，保留设置和数据库结构。

默认产品目录为data/product。所有删除命令都不操作C的formal、natural、training或模型目录。不要将研究目录用作产品数据根目录。

## 5. 已知限制

### 短真人集成检查（环境合适时）

关闭其他Observer运行窗口和占用摄像头的程序；先确认测试环境合适，再运行 `product gui`。任务设置1分钟，正常坐在屏幕前，点击开始；约10秒后暂停，确认摄像头关闭，再继续。可以提前结束，或等待到时结束；点击历史检查记录为已结束、摘要可见。正常会话不应出现离席/手机硬处罚；覆盖不足应诚实显示证据不足。

此轮默认不开截图、不录视频、不要求故意做违规动作。记录窗口是否卡顿、暂停是否释放摄像头、恢复是否成功、结束是否保存。它只验证产品联通性，不替代C盲测或60分钟性能门。

- 一个桌面或控制台运行窗口控制当前会话，尚无托盘、原生通知或跨终端暂停接口；
- 摄像头启动失败不会自动反复重试，可在计划页手动重试；运行中连续约5秒没有新鲜画面时，同一会话自动重连一次。重连期间计为不可观测且不连接前后的违规计时；恢复后继续原会话，失败或再次断流则保存为interrupted、释放摄像头并保留运行窗口；
- 强制退出后，会话在下次run标记interrupted；已经提交的扣分保留，不补算停机时间；
- 聚合观测每秒或有事件时提交，异常崩溃最多丢失末尾约1秒未提交聚合值；
- 提醒在提交后输出控制台，进程恰在二者之间崩溃可能漏显示，持久化事件仍可查；
- 已完成真实图形窗口的手动及定时短闭环；尚未完成带profile确认和60分钟产品负载验证，短测不代表D全部验收通过。
