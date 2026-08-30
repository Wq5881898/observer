# Observer 阶段 C 真实校准与验证运行手册

版本：0.1
日期：2026-08-30
状态：实现交接完成；等待项目所有者参与执行

## 1. 这一步为什么必须有人参与

自动化测试可以证明数据契约、数学计算、版本保护、隐私默认值和状态融合没有明显回归，但不能替代“本人真实看屏幕、倾斜、离座和手持手机”的人工真值。阶段 C 只有在真实 tuning、冻结后的全新 holdout 和 60 分钟正常使用长测完成后才能判断通过。

全程本地运行，不录制视频。校准与验证数值位于 Git 忽略的 `data/`。验证默认只为预测错误保留每 trial 一张关键帧；如果不希望保存任何图像，在 `validate-live` 命令追加 `--no-error-evidence`。

## 2. 测试前固定环境

开始前保持以下条件不变：ThinkPad 内置主屏与内置摄像头、日常座椅高度、正常桌面光照、日常眼镜状态、同一 Windows 电源模式。打开摄像头隐私挡板，退出占用摄像头的软件，并让脸、双眼、双肩和至少一侧手腕进入画面。

在项目目录执行：

```powershell
.\.venv\Scripts\python.exe -m observer doctor
.\.venv\Scripts\python.exe -m observer camera-smoke --seconds 15
.\.venv\Scripts\python.exe -m observer vision-smoke --seconds 30 --models-dir models
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

前三项失败时不要进入校准。冒烟测试不保存画面；没有检测到 Face/Pose 时先检查取景和隐私挡板。

## 3. 建立个人校准档案

```powershell
.\.venv\Scripts\python.exe -m observer calibrate --models-dir models --output data/calibration/default.json
.\.venv\Scripts\python.exe -m observer profile-check data/calibration/default.json
```

校准约需 1–2 分钟。前半段保持可持续的中立坐姿，后半段依次注视 9 个拟合点和 8 个未参与拟合的边缘验证点；边缘归一化 RMSE 必须不高于 0.18，否则档案不保存。按 `Esc` 可安全取消。成功后产生：

- `data/calibration/default.json`：版本化档案、环境指纹、姿势稳健统计和注意力回归系数；
- `data/calibration/default.samples.json`：用于判断复杂设计是否必要的派生数值；不含帧或完整 landmarks。

摄像头索引、实际帧尺寸、主屏尺寸、DPI 或镜像约定变化时，旧档案会拒绝加载并要求重新校准。

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

窗口显示动作后按空格开始。每个批次自动使用独立目录；中途 `Esc` 会把正在执行的 trial 标为 invalid 后退出。`--repetitions 1` 只用于熟悉流程，不能作为验收批次。

每个批次包含：

- `manifest.json`：环境和哈希冻结信息；
- `trials.jsonl`：人工真值、聚合预测、覆盖率、计时与 invalid 原因；
- `derived_samples.jsonl`：用于离线重算的派生数值与原始/确认手机候选，不含图像；
- `report.json`：按能力汇总的混淆矩阵和指标；
- `resources.json`：本批资源统计；
- `evidence/`：只有误判关键帧和到期索引，禁用证据时不存在。

只能依据 tuning 结果调整质量门、特征、阈值或实现。任何调整都记录原因和调整前后指标。

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
