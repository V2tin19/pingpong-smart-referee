"""
adapters —— 各感知模型的适配器实现。

每个适配器负责把某个模型的原始输出翻译成统一的
:class:`core.schemas.FrameObservation`，**不做任何判分**。

当前实现
--------
* :class:`adapters.yolo_adapter.YOLOAdapter` —— YOLO11n 单阶段检测

预留
----
* ``tracknet_adapter.py`` —— 热图峰值/连通域中心，``bbox_xyxy`` 通常为 None
* ``ttnet_adapter.py``    —— 球位置 + 落台/擦网事件分数，需做重复事件合并

导入本包不会加载任何深度学习框架；模型在首次推理时才加载。
"""

from .yolo_adapter import YOLOAdapter

__all__ = ["YOLOAdapter"]
