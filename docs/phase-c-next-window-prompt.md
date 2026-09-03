# Observer 新窗口启动 Prompt

将下面整段复制到新的 Codex 窗口：

```text
请在 C:\Qproject\Observer 继续 Observer 项目的 C 阶段收尾。先不要启动摄像头、不要修改代码、不要调参数、不要提交或清理文件。

请先完整阅读并以此顺序建立上下文：
1. C:\Qproject\Observer\docs\phase-c-window-handoff.md
2. C:\Qproject\Observer\docs\phase-c-empirical-test-report.md（全部历史测试、失败模式和研究结论）
3. C:\Qproject\Observer\docs\phase-c-result.md（重点是第 3 节及记录 34～42）
4. C:\Qproject\Observer\docs\phase-c-formal-test-plan.md（重点是 P0-A-D）
5. C:\Qproject\Observer\docs\phase-c-runbook.md
6. C:\Qproject\Observer\docs\phase-c-validation-handoff.md

这是一个有未提交成果的脏工作树。先检查 git status、关键 diff、当前分支和基准，不得 reset、checkout、覆盖或删除现有修改和测试数据。批次 manifest 的 Git SHA 不能代表其后的未提交改动。不要安装新依赖，也不要重做 17 点校准。

当前停点：D02/D05 的头眼融合修复已经通过两批历史派生数据的离线回放，127 项自动测试通过，但尚未进行修复后的真人验证，所以它仍只是 tuning 候选，不能称为关闭、冻结或 holdout 通过。先复核实现和测试，并运行：
.\.venv\Scripts\python.exe -m unittest discover -s tests -q

唯一明确保留的延期项是 D10：近距离看屏内右下时，眼镜反光可能使虹膜错误。产品语义已经确定：若虹膜因反光不可靠但头部仍朝向屏幕范围，按正常使用处理，不提醒、不扣分，可记低置信度；若头部明确朝屏外则不豁免。但反光识别/降级尚未实现。现在不要为 D10 调阈值、不要安排反光 A/B，也不要把正常戴眼镜的样本判无效；在最终报告中保留为已知限制。

你在检查后先向我简短复述：当前停点、唯一延期项、D01～D07 两轮真人复测的动作和严格标准。不要立即启动，等我明确说“准备好”。随后按交接文档第 7 节的命令，使用完全不变的代码和参数，分别启动两次独立 D01～D07 交互循环。绿色 CAPTURE READY 只代表可采样，不代表模型正确；我先做真实动作并确认，再按空格或回车记录。D02/D05 是头转但眼睛始终看中心；D03/D06 是头和眼睛一起看屏幕边界外，不能把眼睛拉回中心。

每轮结束只保存和分析，不在两轮之间调参。运行时低频静默确认进程正常即可，不逐分钟播报。两轮都需要结合我的动作确认、报告、派生数据和临时计分录像判断。失败先判断是动作无效、不可观测还是算法错误，不要立刻降低阈值。未经我确认，不删除 MP4、接触表、证据图或报告。

若两轮均严格通过，把结果写入 phase-c-result.md，关闭 D02/D05，但继续保留 D10；然后按交接文档第 8 节推进其余 C 阶段门槛。C 阶段仍需连续/holdout、两轮 5 分钟自然测试、60 分钟资源稳定性和复杂度评分，不能在短专项通过后直接进入 D 阶段。

读完和检查完后，只报告你确认到的状态、测试结果和下一次真人复测方案，然后等待我说“准备好”。
```
