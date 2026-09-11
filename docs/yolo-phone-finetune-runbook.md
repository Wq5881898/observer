# Observer 手机检测 YOLO 小规模训练流程

版本日期：2026-09-06

## 1. 目的

本流程用于验证现有 96 张手机标注图片是否足以让 YOLO11n 学会当前的手机边缘场景。

本轮是小规模诊断实验，不是生产模型发布。训练结果必须经过 Observer 的独立评估，不能只看 Ultralytics 自带的 mAP，也不能因为训练集指标很高就直接替换 EfficientDet。

## 2. 当前固定输入

### 数据集

```text
C:\Qproject\Observer\data\annotations\yolo-phone-gt-20260905-s96
C:\Qproject\Observer\data\training\yolo-phone-s96-runholdout
```

当前数据包含 96 张图片：

- 48 张有手机，标签为 `cell_phone`；
- 48 张无手机，使用空标注；
- 数据覆盖底部中央、左右下方和顶部边缘等边界位置；
- 原始 96 张基准数据保持冻结，不在本轮修改。

### 模型和运行环境

```text
预训练权重：C:\Qproject\Observer\yolo11n.pt
训练环境：C:\Qproject\Observer\.yolo-export
设备：CPU
类别：cell_phone（单类别）
```

`.yolo-export` 环境已经包含 Ultralytics 和 PyTorch。本轮不重复安装依赖，也不使用之前的 `.yolo-train` 空环境。

## 3. 训练前检查

在项目根目录执行：

```powershell
cd C:\Qproject\Observer
& .\.yolo-export\Scripts\python.exe -c "import ultralytics, torch; print('ultralytics', ultralytics.__version__); print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
& .\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

训练前应确认：

1. `dataset.yaml` 存在且 `names` 只有 `cell_phone`；
2. 训练和验证图片、YOLO 标签数量对应；
3. 空标注图片仍然存在于数据集；
4. 不修改正式 EfficientDet 模型和正式参数；
5. 训练输出使用新的实验名称，不覆盖旧实验。

## 4. 本轮训练参数

本轮采用保守 CPU 配置，先减少训练时间和增强噪声：

```text
epochs = 12
imgsz = 416
batch = 2
device = cpu
workers = 0
mosaic = 0
mixup = 0
fliplr = 0.5
lr0 = 0.001
patience = 12
skip-val = true
```

说明：

- 关闭 Mosaic 和 Mixup，避免边缘手机被合成到不真实的位置；
- 使用 `skip-val`，避免 CPU 训练期间每一轮验证长时间停顿；
- 训练结束后仍然需要做一次最终验证和 Observer 独立评估；
- 不启用 `single_cls=True`，数据集本身已经是单类别，避免导出元数据出现错误类别名。

## 5. 启动训练

使用新的实验名，例如：

```powershell
cd C:\Qproject\Observer
& .\.yolo-export\Scripts\python.exe .\scripts\train_yolo_phone.py `
  --data .\data\training\yolo-phone-s96-runholdout\dataset.yaml `
  --model .\yolo11n.pt `
  --project .\data\training\runs `
  --name yolo11n-phone-s96-small-20260906 `
  --epochs 12 `
  --imgsz 416 `
  --batch 2 `
  --device cpu `
  --workers 0 `
  --seed 20260906 `
  --mosaic 0 `
  --mixup 0 `
  --fliplr 0.5 `
  --lr0 0.001 `
  --patience 12 `
  --skip-val
```

PowerShell 中每行末尾的反引号 `` ` `` 用于续行。如果复制后出现续行问题，也可以将命令合并为一行。

训练输出目录应为：

```text
C:\Qproject\Observer\data\training\runs\yolo11n-phone-s96-small-20260906
```

主要文件：

```text
weights\best.pt
weights\last.pt
results.csv
args.yaml
```

训练被中断时不要删除输出目录。先保留日志和权重，并记录中断的 epoch；只有明确判定实验无效后才建立新实验名重跑。

## 6. 导出 ONNX

Observer 的 YOLO 后端使用 ONNX。训练结束后，在同一个训练环境中导出：

```powershell
cd C:\Qproject\Observer
& .\.yolo-export\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('data/training/runs/yolo11n-phone-s96-small-20260906/weights/best.pt').export(format='onnx', imgsz=640, device='cpu', nms=False)"
```

导出文件通常位于：

```text
C:\Qproject\Observer\data\training\runs\yolo11n-phone-s96-small-20260906\weights\best.onnx
```

导出后检查模型：

```powershell
& .\.venv\Scripts\python.exe -m observer yolo-doctor --model data\training\runs\yolo11n-phone-s96-small-20260906\weights\best.onnx
```

`ready` 必须为 `true`。模型的 SHA-256 也要记录到实验报告中。

## 7. Observer 独立评估

评估时使用原始 96 张 Pascal VOC 标注，明确单类别名称：

```powershell
cd C:\Qproject\Observer
& .\.venv\Scripts\python.exe -m observer evaluate-phone-annotations `
  --annotations data\annotations\yolo-phone-gt-20260905-s96 `
  --model data\training\runs\yolo11n-phone-s96-small-20260906\weights\best.onnx `
  --output data\model-eval\yolo11n-s96-small-20260906\report.json `
  --thresholds 0.25 0.15 0.10 0.05 `
  --iou-threshold 0.50 `
  --single-class-name cell_phone
```

如果当前 CLI 没有注册 `evaluate-phone-annotations` 子命令，则使用脚本入口：

```powershell
& .\.venv\Scripts\python.exe .\scripts\evaluate_phone_annotations.py `
  --annotations data\annotations\yolo-phone-gt-20260905-s96 `
  --model data\training\runs\yolo11n-phone-s96-small-20260906\weights\best.onnx `
  --output data\model-eval\yolo11n-s96-small-20260906\report.json `
  --thresholds 0.25 0.15 0.10 0.05 `
  --iou-threshold 0.50 `
  --single-class-name cell_phone
```

评估报告至少要比较：

- TP、FN、FP；
- Precision、Recall、F1；
- IoU@0.50；
- 按场景分组的结果；
- 平均 CPU 延迟；
- 与原始 YOLO11n 结果的差异。

## 7.1 Live YOLO 类别名称约定

单类别 YOLO 导出的类别名称来自数据集标签 `cell_phone`，而 MediaPipe
EfficientDet 使用 COCO 名称 `cell phone`。live validation、Vision probe 和
`yolo-replay` 必须统一兼容这两个别名；live validation 创建 YOLO 后端时必须
显式传入 `single_class_name="cell_phone"`。如果遗漏这个参数，单类别 ONNX
输出虽然存在，但解码后的类别名为空，最终会表现为所有手机场景均为
`compliant`，而不是模型真正漏检。

修复后应先用已保存的手机录像运行 `yolo-replay`，确认
`phone_positive_fraction` 大于 0，再进行下一轮摄像头 live validation。

## 8. 结果判定

### 独立性分级

评估报告必须标明属于哪一级，不能统称为“holdout”：

1. 训练集回放：只检查模型是否学会已标注数据；
2. 会话 holdout：不同录像会话，但允许同一手机外观和同一背景；
3. 设备/会话 holdout：手机外观、录像会话均未进入训练，用于最终泛化验收。

当前 P1 属于第 2 级，不属于第 3 级。完整 96 张自定义报告还混合了训练集，
只能用于链路诊断。2026-09-06 的 P2 使用另一种手机外观，属于本轮最有价值
的独立证据；其左右边缘漏检说明当前模型尚不能进入生产或正式替换流程。

### 学习能力检查

如果训练集中的有手机图片不能达到约 95% Recall，优先检查训练链路、标签格式、导出和解码，不要马上增加样本。

### 泛化能力检查

建议目标：

```text
总体 Precision >= 90%
总体 Recall >= 80%
边缘场景 Recall >= 75%（目标 80%）
无手机误报率 <= 2%
```

### 结果解释

| 结果 | 下一步 |
| --- | --- |
| 训练集和验证集都改善 | 进入真实监控回测 |
| 训练集高、验证集低 | 只补充失败位置和距离的样本 |
| 训练集和验证集都低 | 优先检查训练配置和推理链路 |
| Recall 提高但 FP 明显增加 | 增加困难负样本，重新评估阈值 |
| 指标合格但 CPU 延迟恶化 | 不进入正式路径，先做资源优化 |

## 9. 样本扩充规则

本轮不自动增加样本。只有评估显示某个切片持续失败时，才建立独立的下一批目录，例如：

```text
data\annotations\yolo-phone-gt-20260906-hard2-left-bottom
```

每次只补充失败切片，建议 24–48 张有手机和数量匹配的无手机图片。不要把同一段视频的相邻帧随机拆到训练集和验证集。

### 2026-09-06 审计后的下一轮要求

不得在完全相同的 S96 上重复训练。下一轮必须先加入 P2 的新设备困难样本：

- 八段 P2 正例/recovery 录像各抽 6 张，共 48 张；
- 手机正例人工标框，recovery 负例保留空标签并抽查；
- 新数据集版本不得覆盖 S96；
- P2 一旦进入训练，就降级为训练数据，不再作为最终 holdout；
- 训练后另采集 P3 新会话，最好包含不同手机或手机壳，作为设备/会话 holdout。

详细证据和准入条件见 `docs/yolo-phone-model-audit-20260906.md`。

### P2 增量标注包（已生成）

目录：

```text
C:\Qproject\Observer\data\annotations\yolo-phone-gt-20260907-p2hard-s48
```

来源是 P2 的八段 trial video，每段均匀抽取 6 帧，共 48 张：四段手机候选
录像 24 张，四段 recovery 录像 24 张。`source_run` 已正确记录为 P2。

人工标注时：

1. Open Dir 选择该目录下的 `images`；
2. Change Save Dir 选择 `labels-voc`；
3. 输出格式可使用 PascalVOC 或 YOLO，标签只使用 `cell_phone`；同一个标注包
   内应保持一种格式；
4. 只框实际可见的手机部分，边缘截断时框贴住图片边界；
5. 即使文件来自手机候选录像，也必须以当前图片实际内容为准；没有手机就不框；
6. recovery 图片原则上无手机，但仍需人工抽查，不根据场景名称盲目生成标签；
7. 不参考模型预测结果，避免把旧模型偏差写入真值。

用户完成所有有手机图片标注并确认剩余图片均无手机后，才运行
`fill_confirmed_empty_phone_annotations.py` 生成同格式空标签。随后使用支持多标注根的
`prepare_yolo_phone_finetune.py` 生成新数据版本，计划划分为 P0+P2 train、
P1 val。生成命令必须使用新的输出目录，不能覆盖 S96。

本轮实际采用 YOLO TXT，数据生成命令为：

```powershell
& .\.venv\Scripts\python.exe .\scripts\prepare_yolo_phone_finetune.py `
  --annotations data\annotations\yolo-phone-gt-20260905-s96 `
  --additional-annotations data\annotations\yolo-phone-gt-20260907-p2hard-s48 `
  --output-root data\training\yolo-phone-s96-p2hard-runholdout `
  --train-runs P0,P2 `
  --val-runs P1
```

训练结果和后续 P3 要求见 `docs/yolo-phone-training-p2hard-20260907.md`。

## 10. 交接记录

每次训练必须记录：

1. 数据集路径和样本数量；
2. 训练命令和完整参数；
3. 预训练权重 SHA-256；
4. 训练输出目录；
5. ONNX 文件和 SHA-256；
6. Observer 评估报告路径；
7. 每个阈值的 TP/FN/FP/Precision/Recall/F1；
8. CPU 延迟和内存观察；
9. 是否允许进入真人回测；
10. 是否允许替换正式 EfficientDet。

正式 EfficientDet 路径在 YOLO 通过独立 holdout、资源和误报检查之前保持不变。

## 11. 联合能力保留与快速回归（2026-09-08）

边缘手机微调后必须同时回归完整手机和非手机物体。详细实验、数据来源和候选模型见：

```text
docs/yolo-phone-joint-retention-20260908.md
```

常规持续场景回归统一使用 16 个均匀帧，避免每次处理整段 100～155 帧：

```powershell
& .\.venv\Scripts\python.exe -m observer yolo-replay `
  --video <video.mp4> `
  --model <best.onnx> `
  --output <report.json> `
  --score-threshold 0.25 `
  --sample-frames 16
```

只有研究瞬时漏检、持续时间或 retain 桥接机制时才省略 `--sample-frames` 做逐帧回放。任何训练后回归至少包含：完整单手、完整双手、四类边缘位置、各恢复段、自然无手机和手持非手机物体。

### 短程增量训练的 warm-up 规则

Ultralytics 的默认 `warmup_epochs=3`、`warmup_bias_lr=0.1` 不适合只有 2–6 个
epoch 的小样本增量训练。名义上的低 `lr0` 并不能限制 warm-up 期间的 bias
学习率，可能在一轮内同时提高召回并破坏困难负例。短程微调必须在命令中明确记录
并通常使用：

```powershell
--lr0 0.00001 --warmup-epochs 0 --warmup-bias-lr 0
```

如确需 warm-up，必须从训练日志核对各参数组实际学习率，不能只引用 `lr0`。
训练脚本现已提供这两个显式参数。每个 checkpoint 仍要分别做历史正例、负例和
目标困难场景回放；不得直接采用 Ultralytics 自动选择的 `best.pt`。

## 12. B06 轻量补强（2026-09-08）

P3b 真人验收中，白色手机在双手、低位、画面左下条件下最终能够触发
`violation`，但违规帧占比约 63.5%，低于严格连续性验收线。该问题按“小范围连续性
补强”处理，不重新训练全部模型，也不通过大幅降低阈值绕过。

人工标注包（已完成 24/24）：

```text
data/annotations/yolo-phone-b06-light-s24-20260908
```

该包从 P3b 的 B06 录像中只筛选 24 张当前模型低置信度帧。每张图片均明确包含
同一部白色手机，最终使用 YOLO 文本格式标注 `cell_phone`（历史目录名仍为
`labels-voc`），框只覆盖手机可见部分，不得包含双手或猜测被遮挡区域。原录像在
参与训练后不再作为独立验收证据。

这 24 张难正例已与 S216 联合数据混合；从 S216 `best.pt` 继续训练，冻结骨干网络，
完成 6 epochs 低学习率微调。逐检查点比较后选定 `weights/epoch5.onnx` 进入真人
holdout：历史正例 112/112 检出，历史负例 1/112 瞬时误报，独立 B03 为 0/16。
旧 B06 全录像 137/137 只证明拟合改善，因为它已经参与训练。下一次重新拍摄的 B06
必须作为 holdout，仍按至少 90% violation、最长连续至少 10 秒的冻结标准判定。
完整结果见 `yolo-phone-b06-light-finetune-20260908.md`。

极低下边缘专项使用独立来源 `BOTTOM_EDGE_TRAIN`，标注包为
`data/annotations/yolo-phone-bottom-edge-s16-20260909`。该包只有 16 张代表帧，
使用 YOLO 格式；框只覆盖手机在画面内的可见部分，并允许框接触画面底边。它不得
与标准 B06 混为同一验收语义，训练后必须分别回归自然低位和极低露边。

### 充电宝困难负例门禁

手持充电宝与手机在单帧中的矩形轮廓、尺寸和手部关系高度相似，不能把“加入一段
空标签录像”视为必然能够解决误报。2026-09-09 的 S272 极轻量试验中，16 张
充电宝空标签经过 2 epochs 后仍有 15/16 至 16/16 帧被判手机，同时极低下边缘
手机从 15/16 下降到 13/16；该分支已淘汰。以后针对充电宝等相似物体训练时：

1. 同一录像进入训练后只能做拟合检查，必须另拍新角度、新距离作为 holdout；
2. 每个检查点必须同时回归完整手机、边缘手机、恢复段和历史负例；
3. 如果困难负例没有明显下降而正例召回开始下降，应立即停止，不得继续堆叠轮次；
4. 单一物体、单一背景和单一手持轨迹不足以证明非手机类别的泛化能力。

详细记录见 `docs/yolo-phone-b06-light-finetune-20260908.md` 第 10 节。

### S304 新录像复测（P7）

冻结候选为 `data/training/runs/yolo11n-phone-s304-contrast-e2b-nowarm-20260909/weights/epoch1.onnx`。
使用 E01、E01 recovery 和 F00–F09，程序按目录中的场景顺序先执行 E 再执行 F；
总记录时长 100 秒，另加人工准备时间。输出到 `data/formal/phone-contrast/P7-s304-manual`。
沿用交互式确认，真实动作已完成但模型仍黄色时也应记录；不得为变绿调整露出比例。
E01 尽量复现上轮露出比例，F 配对物体保持相同位置、角度和可见比例。

本批 CLI 使用 `--role tuning`，因为现有 E/F 用例只允许诊断角色；应称为冻结模型的
新录像复测，不冒称正式 holdout 协议。结果审阅前不用于训练，按既有用例标准判定。
原始漏检间隔小于 retain 仅支持桥接假设，不能证明实际追踪、手持规则和事件连续性
必然通过。过去按 MP4 的 15fps 计算的间隔也不等同于采集墙钟时间，应优先核对
录像 timestamps sidecar 与 live report。正式结论以本轮真实规则输出和人工复核为准。

### F 组：手机/充电宝成对采集

S272 的单段充电宝空标签轻量训练没有形成有效区分，因此下一轮不继续使用同一
16 帧堆叠训练。新增 F00-F09 独立诊断组，用同一背景、同一手、近似相同的位置、
角度和露出比例交替拍摄充电宝与手机。每个充电宝负例放在对应手机正例之前，并在
位置切换之间插入 5 秒空手清除段，避免 1.5 秒候选保留污染负例开头。

本组只测试中央、左下和右下三个位置；物体至少露出三分之一，不测试视觉上不可
区分的极小碎片。录像保存后，每段只选少量代表帧，充电宝使用受控空标签，手机框
只覆盖实际可见部分。F 组属于训练数据采集和诊断，不改变 B/E 组历史基线，也不得
被重新命名为独立 holdout。

手机专项界面（包括 B/E/F 组）同时显示两个状态：`CAPTURE QUALITY` 只说明
画面、脸部和距离是否可录制；大号 `TARGET READY` 还要求当前模型状态符合该
用例，并稳定 0.6 秒。
撤销动作后大号标记应恢复为黄色；手机候选有最长 1.5 秒的保留期，因此颜色可能
短暂延迟，但不得永久保持绿色。充电宝当前很可能显示
`MODEL TARGET: NOT MATCH`/`TARGET NOT READY`，这是需要保留的失败证据；确认
真实动作和采集质量正确后仍可按空格记录，不要为了迎合模型改变动作。注意力和
姿态等非手机协议仍保留原有的质量就绪徽标，避免把模型输出当作注视或姿态真值。

```powershell
& .\.venv\Scripts\python.exe -m observer validate-live `
  --models-dir models `
  --profile data/calibration/default.json `
  --role tuning `
  --signals phone_held `
  --scenarios phone_contrast_f00_clear,phone_contrast_f01_powerbank_center,phone_contrast_f02_phone_center,phone_contrast_f03_clear,phone_contrast_f04_powerbank_left,phone_contrast_f05_phone_left,phone_contrast_f06_clear,phone_contrast_f07_powerbank_right,phone_contrast_f08_phone_right,phone_contrast_f09_final_clear `
  --output-root data/formal/phone-contrast/P6-paired-s256-manual `
  --debug-trial-videos `
  --phone-score-threshold 0.25 `
  --phone-retain-seconds 1.5 `
  --phone-center-distance 0.18 `
  --detector-backend yolo `
  --yolo-model data/training/runs/yolo11n-phone-s256-bottomedge-e2b-nowarm-20260909/weights/epoch1.onnx
```
