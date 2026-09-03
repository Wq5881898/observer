# Observer 阶段 C：窗口交接状态

日期：2026-09-03  
工作目录：`C:\Qproject\Observer`  
阶段状态：**C 阶段尚未完成；D02/D05 专项已由两轮真人复测关闭；下一步是 P0-B/P0-C 两轮交互基线**  
分支/已知基准提交：`main` / `c2be3a1ed86c5de9e5441af08aea73d5cd9e33ce`

## 1. 新窗口先遵守的边界

- 不进入 D 阶段，不把当前结果称为最终训练完成或产品准确率。
- 工作树包含尚未提交的 C 阶段代码、文档和测试；这些都是当前工作成果。不得 `reset`、`checkout`、清理或覆盖。
- 当前批次清单中的 Git SHA 只表示运行时的已提交基线，不能代表其后的未提交修改。
- 先阅读本文件及第 2 节列出的资料、检查差异并运行自动测试；不要一进入新窗口就启动摄像头或继续调参。
- 调试 MP4 是经用户允许的临时人工核验资料。未经用户确认，不删除历史录像、接触表、报告或派生数据。
- 不安装新依赖，不重新训练 Face、Pose、EfficientDet，也不重做 17 点校准，除非后续证据明确证明环境或校准已失效。

## 2. 资料读取顺序

1. `docs/phase-c-window-handoff.md`（本文件，当前状态的最高优先级摘要）
2. `docs/phase-c-result.md`，重点阅读第 3 节及记录 34～43
3. `docs/phase-c-formal-test-plan.md`，重点阅读 P0-A-D 与交互/连续测试边界
4. `docs/phase-c-runbook.md`
5. `docs/phase-c-validation-handoff.md`（完整阶段验收和复杂度评分框架）

阶段 C 截至当前的论文式测试总报告见 `docs/phase-c-empirical-test-report.md`；它用于回顾全部测试、失败模式和方法改进，不替代上述当前停点顺序。

若旧记录与本文件冲突，以日期更晚的 `phase-c-result.md` 第 41～43 条和本文件为准。特别是第 40 条曾提出反光 A/B 测试，第 41 条已经明确取代该安排：当前不再计划反光专项测试。

## 3. 已经确认的测试方法

- 基线/调优使用交互模式：界面先给动作要求；绿色 `CAPTURE READY` 只表示画面质量、脸/眼可见性及指定距离适合采样，不表示模型已经判断正确。
- 测试者先完成真实动作并自行确认，再按空格或回车；1 秒准备后才开始计分，计分期间隐藏模型结论，避免为了“迎合绿色”扭曲动作。
- 交互基线通过后，才能做无确认、无模型反馈的连续循环和自然使用测试。
- 同一缺陷的复现优先用两次独立短循环；大量误判或严重漏检才做专项边界测试。已有稳定能力不因每次小参数调整重复全量真人测试。
- 近距、自然距离和远距都需要代表性验证，但 17 点坐标校准仍有效。只有某个距离的代表点失败，才升级为该距离的专项网格，而不是重新测试全部 17 点。

## 4. 当前技术与产品结论

### 4.1 已确认并保留

- 完整脸和双眼可见即可进行注意力观察；肩膀不足不阻止注意力和在席判断。
- 姿态/人体工学与注意力分开：肩部不平、身体倾斜、歪头、过度低头、异常头眼补偿可以作为健康观察，但不自动等同于效率违规或重复扣分。
- 注意力的左右正例必须是头和眼睛共同看向屏幕外；“头转向一侧、眼睛补偿看回屏幕中心”应为注意力合规。
- 水平侧看不能冒充向下看；低头要求方向一致的头姿与虹膜证据。
- 摄像头被占用、挡板关闭、镜头被肉色/棕色手指遮挡等都必须给出可理解的不可用/不可观测状态，不能黑屏静默退出或伪装成离席。
- 默认不保存连续视频；只保留评分、原因和策略允许的关键证据。临时录像只用于当前调试，核验后由用户决定清理。

### 4.2 唯一明确延期项：D10 近距右下/眼镜反光

项目所有者已确认以下产品语义：

- 当虹膜因眼镜反光不可可靠判断，而头部仍朝向屏幕范围时，按正常使用处理：不提醒、不扣分；允许在本地记录低置信度供以后学习。
- 如果头部本身明确朝向屏幕外，不能用反光豁免。

目前尚未实现可靠的反光识别或上述降级逻辑。因此，D10“近距离看屏内右下”仍是已知限制，也是这次交接时唯一明确保留的未关闭项。不要为了关闭它继续调阈值、安排反光 A/B 或把测试者正常戴眼镜视为无效输入；留待后续单独设计。

## 5. D02/D05 头眼融合修复：真人复测已关闭

代码候选位于 `src/observer/calibration.py`：

- 仅当同会话相对偏航 `abs(yaw_delta) >= 0.30` 时启用；
- 用 `screen_x - 2.0 * yaw_delta` 校正虹膜横向屏幕坐标；
- 有虹膜结果时，不再让大幅头转单独覆盖眼睛补偿；
- 虹膜不可观测时仍保留明确头转的安全兜底；
- 小幅头动、垂直规则和全局边界没有改变。

报告器也已改为按严格标准判断，而不是只看 trial 聚合标签；新 trial 会记录各状态最长连续时长。

自动测试结果：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

最近结果为 `Ran 127 tests ... OK`。

两批历史特征的离线回放支持该候选：

| 场景 | 修复前主要问题 | 候选离线回放 |
| --- | --- | --- |
| D02 头左、眼看中心 | 第二轮 47.1% unknown | 98.3% compliant、0 unknown |
| D05 头右、眼看中心 | 两轮 89.4%/100% violation | 两轮约 98.2% compliant，最长 violation 为 0 |
| D03 真正屏外左看 | 有效轮应为正例 | 100% violation，最长 10.00 秒 |
| D06 真正屏外右看 | 两轮均为正例 | 100% violation，最长约 9.92 秒 |
| D10 近距屏内右下 | 可复现误判 | 仍为 100%/91.2% violation，未被掩盖 |

两轮独立真人交互循环已按第 7 节原命令完成：`tuning-20260903T071027Z-5fbe7734` 和 `tuning-20260903T071636Z-7243f59f`。报告、派生数据、抽样录像帧和测试者动作确认均支持 D01～D07 严格通过。据此关闭 D02/D05 头眼补偿误报专项；这不代表注意力整体冻结、holdout 通过或阶段 C 完成。

## 6. 关键批次和证据

- 首轮 P0-A-D：`data/formal/attention-compensation-interactive/tuning-20260903T042922Z-c524138e`
  - D03 的实际动作是重复 D02（头左、眼看中心），所以该 D03 标签无效；不得当作左侧漏检。
  - D05 与 D10 是经录像确认的真实误判。
- 第二轮 P0-A-D：`data/formal/attention-compensation-interactive/tuning-20260903T045656Z-cbd28f79`
  - D03、D06 均为有效屏外正例并持续通过。
  - 用户人工确认 D05 的眼睛始终看中心、D10 确实看屏内右下。
  - 报告在该批次的 `report.json`；完整计分录像和接触表位于批次目录下。
- 修复后第 1 轮：`data/formal/attention-compensation-postfix/tuning-20260903T071027Z-5fbe7734`
- 修复后第 2 轮：`data/formal/attention-compensation-postfix/tuning-20260903T071636Z-7243f59f`
- 两轮的抽样审阅帧分别在批次内 `_review-frames-cycle1` 和 `_review-frames-cycle2`；这些和 MP4、接触表、报告都是待项目所有者决定清理的临时证据。
- 当前个人校准：`data/calibration/default.json`
  - `profile_id = dc9476b1-f092-4864-b661-665fa9008483`

## 7. 当前下一项执行任务

P0-A-D 已完成。下一步对正式 P0-B/P0-C 各做两次重新启动的交互循环；轮间只保存和分析，不改代码、不改参数。实现拆分如下：B08 的 phone 与 attention 分别采样，C06 的 posture 与 attention 分别采样，测试者重复保持同一真实动作，避免当前单主信号 trial 架构伪造同时判定。文字 overlay 已改为小字号并按画面宽度折行。

每个循环前运行 131 项自动测试并确认工作树无意外变化，等待项目所有者明确说“准备好”后再启动摄像头。P0-B 输出根目录用 `data/formal/phone-baseline-postfix`；P0-C 输出根目录用 `data/formal/posture-ergonomics-baseline-postfix`。场景 ID 和严格标准见 `docs/phase-c-formal-test-plan.md` 与 `src/observer/live_validation.py`；报告器会逐场景执行这些标准。

P0-B 每轮使用：

```powershell
.\.venv\Scripts\python.exe -m observer validate-live --models-dir models --profile data/calibration/default.json --role tuning --signals phone_held,attention_away --scenarios phone_baseline_b01_natural_no_phone,phone_baseline_b02_desktop_phone,phone_baseline_b03_non_phone_object,phone_baseline_b04_one_hand_phone,phone_baseline_b05_recovery,phone_baseline_b06_two_hands_low,phone_baseline_b07_recovery,phone_baseline_b08_phone_out_of_frame,attention_baseline_b08_look_off_screen,phone_baseline_b09_final --output-root data/formal/phone-baseline-postfix --debug-trial-videos
```

P0-C 每轮使用：

```powershell
.\.venv\Scripts\python.exe -m observer validate-live --models-dir models --profile data/calibration/default.json --role tuning --signals posture_deviation,attention_away --scenarios posture_ergonomics_c01_natural,posture_ergonomics_c02_near_natural,posture_ergonomics_c03_forward_slouch,posture_ergonomics_c04_recovery,posture_ergonomics_c05_side_or_tilt,posture_ergonomics_c06_shoulder_unknown,attention_ergonomics_c06_attention,attention_ergonomics_c07_small_chin,posture_ergonomics_c08_final --output-root data/formal/posture-ergonomics-baseline-postfix --debug-trial-videos
```

两轮都要由测试者确认动作是否真实有效。失败时先核验录像和派生数据，区分动作无效、反光/不可观测与算法错误；不要立即继续改阈值。运行期间只需低频静默确认进程仍在运行，完成后再集中分析，避免无意义的逐分钟文字播报。

如果任一轮失败：先结合测试者确认、报告、派生数据和临时录像区分动作无效、不可观测和算法错误，保留原始批次，再决定是否修复。

## 8. 完成头眼复测后的 C 阶段剩余工作

不要因为 D01～D07 通过就宣布 C 阶段完成。后续仍须按正式计划执行并形成证据：

1. 对尚未冻结的注意力、手机、姿态/人体工学做规定的交互基线回归；D10 按已批准延期处理，不阻塞头眼候选关闭，但必须在最终限制中披露。
2. 基线通过后，冻结代码和配置，再做无模型反馈的连续循环/holdout；调过参数的数据不能回称 holdout。
3. 至少两轮 5 分钟真实自然使用，重点统计误报、漏检、unknown、事件时长及关键证据。
4. 做阶段要求的 60 分钟资源与稳定性测试；短测约 CPU 平均 7%～8%、P95 约 8%～10%、内存峰值约 290 MB，只是基线，不是长测结论。
5. 完成 `docs/phase-c-validation-handoff.md` 中的复杂度评分，明确 keep/simplify/remove/defer；特别评估曾担心的过度设计。
6. 汇总最终 C 阶段报告、已知限制、隐私证据清理清单，经项目所有者确认后才进入 D 阶段。

## 9. 工作树交接提醒

截至交接，工作树有多项 C 阶段修改及新文件，包含校准、融合、验证器、规则、CLI、文档、测试和调试提取脚本。新窗口必须以实际 `git status --short` 与 `git diff` 为准，不依赖本文枚举判断文件是否可以删除。清理数据、覆盖产物或继续提交推送都必须逐次获得项目所有者确认。
