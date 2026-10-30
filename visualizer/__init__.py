"""
visualizer —— 转播风格可视化。

* :class:`visualizer.broadcast_hud.BroadcastHUD` —— 比分板 + 轨迹尾迹 + 2D 俯视雷达

导入本包不加载任何深度学习框架；只依赖 numpy / opencv。
"""

from .broadcast_hud import BroadcastHUD, HUDState

__all__ = ["BroadcastHUD", "HUDState"]
