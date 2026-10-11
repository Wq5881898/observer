# Deliberate allowlist: never collect the repository or user data wholesale.
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

root = Path(SPECPATH).parent
stage = Path(os.environ['OBSERVER_BUILD_STAGE'])
model_files = ['manifest.json', 'face_landmarker.task', 'pose_landmarker_lite.task',
               'efficientdet_lite0_int8.tflite', 'yolo/observer-phone-s304-e2b.onnx']
datas = [(str(root / 'models' / name), str(Path('models') / Path(name).parent))
         for name in model_files]
datas += [(str(root / 'src/observer/default_config.json'), 'observer'),
          (str(stage / 'build-info.json'), '.'),
          (str(stage / 'licenses'), 'licenses')]
datas += collect_data_files('mediapipe')
datas += copy_metadata('mediapipe') + copy_metadata('onnxruntime')
a = Analysis([str(root / 'packaging/entry.py')],
    pathex=[str(root / 'src'), str(root / 'packaging')],
    binaries=collect_dynamic_libs('mediapipe') + collect_dynamic_libs('onnxruntime'),
    datas=datas, hiddenimports=['frozen_diagnostics', 'PySide6.QtNetwork', 'tzdata'],
    hookspath=[], runtime_hooks=[],
    excludes=['jax', 'jaxlib', 'torch', 'tensorflow', 'IPython', 'pytest'],
    hooksconfig={'matplotlib': {'backends': ['Agg']}}, noarchive=False)
pyz = PYZ(a.pure)
diagnostic = EXE(pyz, a.scripts, [], exclude_binaries=True, name='Observer-diagnostic',
                 console=True, debug=False, strip=False, upx=False)
gui = EXE(pyz, a.scripts, [], exclude_binaries=True, name='Observer',
          console=False, debug=False, strip=False, upx=False)
COLLECT(diagnostic, gui, a.binaries, a.datas, name='Observer', strip=False, upx=False)
