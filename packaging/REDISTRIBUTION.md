# 内部候选的许可清单与未审查项

该候选没有获得公开分发结论。`_internal/licenses/inventory.json` 记录构建环境的实际版本、包自报许可及可发现的许可文件；其中包含可能未进入运行包的构建依赖，不等于法律审查或完整 SBOM。

必须另行核对：Python 及原生运行库、PySide6/Qt 的 LGPL/GPL/商业许可义务、MediaPipe/TFLite 及 Face/Pose/EfficientDet 模型各自分发条件、ONNX Runtime、OpenCV、NumPy/SciPy 及其二进制依赖的通知要求。JAX、PyTorch、TensorFlow 在此候选中排除；运行加载验证决定是否满足本项目实际使用路径。

冻结手机 ONNX 来自项目 YOLO11n 微调模型；Ultralytics 的 AGPL/Enterprise 条件及模型来源/训练数据权利尚待审查，转换 ONNX 不自动免除原许可义务。manifest 内模型下载地址是来源记录，不是许可证。不得以第三方文件已收集作为可公开分发证明。

E 已通过并关闭。公开发布仍受 F 人工／独立环境验收、手机识别缺陷、许可材料和项目所有者签收约束。

2026-10-25：所有者已选择 AGPL 开源路线，项目源码条款见根目录 LICENSE／NOTICE；GPLv3 与 LGPLv3 正文已补入。构建收集器现包含 ThirdPartyNotices、COPYRIGHT 并记录逐文件摘要，已用实际构建环境验证 ONNX Runtime 第三方通知原字节收集。此修复只作用于未来构建，原 f008 未修改。

源码交付要求及尚未完成的模型／第三方范围见 SOURCE-DELIVERY.txt。声明源码许可不替代接收者实际获取对应源码的安排。
