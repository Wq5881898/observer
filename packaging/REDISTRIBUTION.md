# 内部候选的许可清单与未审查项

该候选没有获得公开分发结论。`_internal/licenses/inventory.json` 记录构建环境的实际版本、包自报许可及可发现的许可文件；其中包含可能未进入运行包的构建依赖，不等于法律审查或完整 SBOM。

必须另行核对：Python 及原生运行库、PySide6/Qt 的 LGPL/GPL/商业许可义务、MediaPipe/TFLite 及 Face/Pose/EfficientDet 模型各自分发条件、ONNX Runtime、OpenCV、NumPy/SciPy 及其二进制依赖的通知要求。JAX、PyTorch、TensorFlow 在此候选中排除；运行加载验证决定是否满足本项目实际使用路径。

冻结手机 ONNX 来自项目 YOLO11n 微调模型；Ultralytics 的 AGPL/Enterprise 条件及模型来源/训练数据权利尚待审查，转换 ONNX 不自动免除原许可义务。manifest 内模型下载地址是来源记录，不是许可证。不得以第三方文件已收集作为可公开分发证明。

公开发布仍受 E 遗留门、打包验收和项目所有者签收约束。
