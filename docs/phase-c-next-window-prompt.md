# Observer 新窗口启动 Prompt

将下面整段复制到新的 Codex 窗口：

```text
请在 C:\Qproject\Observer 接手 Observer 项目。阶段 C 已通过并关闭。不要重复
C 阶段摄像头测试，不要训练或调参数，也不要提交、删除或重置任何文件。

按顺序完整阅读：
1. C:\Qproject\Observer\docs\phase-c-window-handoff.md
2. C:\Qproject\Observer\docs\phase-c-closeout-audit-20260909.md
3. C:\Qproject\Observer\docs\phase-c-complexity-review-20260910.md
4. C:\Qproject\Observer\docs\phase-c-result.md 的“2026-09-10 当前结论”
5. C:\Qproject\Observer\docs\phase-c-formal-test-plan.md

这是包含大量已完成成果的脏工作树。先执行 git status 和关键 diff，保护用户已有
修改，不得 reset、checkout 或清理。随后运行：
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\uv.exe lock --check --cache-dir .uv-cache
.\.venv\Scripts\python.exe -m observer doctor
.\.venv\Scripts\python.exe -m observer yolo-doctor --model models\yolo\observer-phone-s304-e2b.onnx

当前冻结方案：YOLO S304 单类模型检测手机，低频 EfficientDet 只补充 person；
在席/离席和清晰持续手机是硬信号；attention、姿态和人体工学为零效率扣分软
信号；不可观测必须为 unknown。模型 SHA-256 必须是
C4A8A791E227780B334E075D671D4347ED9D5FEE7474D6F017659366709C486F。

两轮5分钟自然使用、离席/手机事件链和最终144秒最小盲测已经完成。最终盲测批次
`20260910-221353-3e408b17` 自动检查12/12通过并完成人工真值复核；首轮失败批次
必须保留。手机跟踪已修复远端假框劫持，候选保留为2.0秒。当前全仓库207项自动
测试通过。

C阶段最终结论是通过并关闭。60分钟无录像长测批次
`data/natural/c-closeout/20260910-234159-098b5ab5` 已完成：45,615帧、零读取失败，
CPU平均12.51%、P95 16.79%、内存峰值403.92 MB、增长18.61 MB/小时，预算通过；
硬信号没有形成与证据冲突的规则级误报。不得为取得更漂亮指标重复该长测。

极低露边手机、相似物体短误框、眼镜反光、attention会话波动、跨设备泛化和精细
人体工学已经列为非阻塞限制。不继续YOLO补丁训练，不新增ROI，不重建attention，
不重做17点校准，也不恢复大规模测试矩阵。失败时只分析并补对应最小场景。

读完后向我报告：工作树是否完整、测试/锁/模型哈希是否通过，以及准备继续的下一
阶段具体范围。不要重新打开阶段 C 已延期的边缘研究。
```
