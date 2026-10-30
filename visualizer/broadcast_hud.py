"""
visualizer/broadcast_hud.py
===========================

转播风格 HUD：比分板 + 轨迹尾迹 + 2D 俯视雷达。

组长要求：**视频左上角显示比分**。因此比分板固定在左上角，字号最大。

设计上刻意与老脚本 ``smart_referee_yolo.py`` 的一处做法不同
----------------------------------------------------------
老脚本每帧都把**空中球**用单应性投影到 2D 雷达上。组长方案 §5 明确
"单目摄像头不能把空中球的位置直接通过桌面透视矩阵转换成真实落点"——
投影出来的点会落在台面上，视觉上等于宣称"球在台面这里"，是误导。

本实现**只在落台瞬间**往雷达上打点，雷达上出现的每个点都有真实落台事件支撑。
球在空中的位置只用图像上的检测点表示，不投影。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.geometry import TABLE_LENGTH_CM, TABLE_WIDTH_CM, TableCalibration
from core.homography import LandingPoint, TableHomography
from core.trajectory_processor import MotionState, TrackPoint

__all__ = ["HUDState", "BroadcastHUD"]

# 参考分辨率：HUD 的所有尺寸都按帧宽相对它缩放
_REF_W = 1920.0

# 颜色（BGR）
_C_BG = (22, 22, 22)
_C_BORDER = (95, 95, 95)
_C_TEXT = (245, 245, 245)
_C_DIM = (150, 150, 150)
_C_P1 = (255, 210, 60)      # 左方（画面左侧）
_C_P2 = (90, 230, 120)      # 右方
_C_SERVER = (60, 255, 60)
_C_MEASURED = (60, 235, 60)
_C_SMOOTHED = (255, 210, 60)
_C_PREDICTED = (60, 165, 255)
_C_LANDING_IN = (60, 235, 60)
_C_LANDING_OUT = (60, 60, 255)
_C_EVENT = (0, 235, 255)


@dataclass
class HUDState:
    """渲染一帧所需的一切。由调用方（runner）组装。"""

    frame_id: int = 0
    timestamp_s: float = 0.0
    fps: float = 0.0
    score: Tuple[int, int] = (0, 0)
    state: str = "WAITING_SERVE"
    server: str = "A"
    game_over: bool = False
    last_decision: Optional[str] = None
    decision_age_s: float = 999.0
    track_points: List[TrackPoint] = field(default_factory=list)
    landings: List[LandingPoint] = field(default_factory=list)
    current_pos: Optional[Tuple[float, float]] = None
    track_method: str = "none"
    visible: bool = True


class BroadcastHUD:
    """把状态机结果与轨迹画到帧上。

    Parameters
    ----------
    calibration:
        球台标定（用于画台面轮廓与球网）。
    homography:
        单应性（用于把落点画到 2D 雷达上）。
    trajectory_gap_px / trajectory_gap_frames:
        分段断连阈值：超过就断开线段，绝不跨屏连假线。
    landing_history:
        雷达与画面上保留多少个最近落点。
    """

    def __init__(
        self,
        calibration: TableCalibration,
        homography: TableHomography,
        trajectory_gap_px: float = 85.0,
        trajectory_gap_frames: int = 2,
        landing_history: int = 6,
    ) -> None:
        self.calibration = calibration
        self.homography = homography
        self.trajectory_gap_px = float(trajectory_gap_px)
        self.trajectory_gap_frames = int(trajectory_gap_frames)
        self.landing_history = int(landing_history)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def render(self, frame: np.ndarray, hud: HUDState) -> np.ndarray:
        """在 frame 上原地绘制 HUD，并返回它。"""
        if frame is None or not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy array (BGR image)")
        self._s = float(frame.shape[1]) / _REF_W
        self._draw_table(frame)
        self._draw_trajectory(frame, hud)
        self._draw_ball(frame, hud)
        self._draw_landings(frame, hud)
        self._draw_scoreboard(frame, hud)       # 组长要求：左上角比分
        self._draw_radar(frame, hud)
        return frame

    # ------------------------------------------------------------------
    # 缩放工具
    # ------------------------------------------------------------------
    def _px(self, v: float) -> int:
        return max(1, int(round(v * self._s)))

    def _put(
        self,
        img: np.ndarray,
        text: str,
        org: Tuple[int, int],
        scale: float,
        color: Tuple[int, int, int],
        thick: int = 1,
    ) -> None:
        cv2.putText(
            img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale * self._s, color,
            max(1, int(round(thick * self._s))), cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # 球台与球网
    # ------------------------------------------------------------------
    def _draw_table(self, img: np.ndarray) -> None:
        poly = np.asarray(self.calibration.table_polygon, dtype=np.int32)
        cv2.polylines(img, [poly], True, (255, 220, 0), self._px(2), cv2.LINE_AA)
        y0 = int(min(p[1] for p in self.calibration.table_polygon))
        y1 = int(max(p[1] for p in self.calibration.table_polygon))
        cv2.line(
            img, (self.calibration.net_x, y0), (self.calibration.net_x, y1),
            (255, 0, 255), self._px(2), cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # 轨迹：分段绘制，绝不跨屏连假线
    # ------------------------------------------------------------------
    def _draw_trajectory(self, img: np.ndarray, hud: HUDState) -> None:
        pts = hud.track_points
        if len(pts) < 2:
            return

        segments: List[List[TrackPoint]] = []
        cur = [pts[0]]
        for p in pts[1:]:
            prev = cur[-1]
            df = abs(p.frame_id - prev.frame_id)
            dd = float(np.hypot(p.x - prev.x, p.y - prev.y))
            if df > self.trajectory_gap_frames or dd > self.trajectory_gap_px:
                segments.append(cur)
                cur = [p]
            else:
                cur.append(p)
        segments.append(cur)

        for seg in segments:
            if len(seg) < 2:
                continue
            for a, b in zip(seg[:-1], seg[1:]):
                if a.state == MotionState.PREDICTED or b.state == MotionState.PREDICTED:
                    color, thick, alpha = _C_PREDICTED, 1, 0.45
                elif a.state == MotionState.SMOOTHED or b.state == MotionState.SMOOTHED:
                    color, thick, alpha = _C_SMOOTHED, 2, 0.9
                else:
                    color, thick, alpha = _C_MEASURED, 2, 1.0
                self._line(img, (a.x, a.y), (b.x, b.y), color, thick, alpha)

    def _line(
        self,
        img: np.ndarray,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        color: Tuple[int, int, int],
        thick: int,
        alpha: float,
    ) -> None:
        pt0 = (int(round(p0[0])), int(round(p0[1])))
        pt1 = (int(round(p1[0])), int(round(p1[1])))
        if alpha >= 1.0:
            cv2.line(img, pt0, pt1, color, self._px(thick + 3), cv2.LINE_AA)
            cv2.line(img, pt0, pt1, color, self._px(thick), cv2.LINE_AA)
            return
        overlay = img.copy()
        cv2.line(overlay, pt0, pt1, color, self._px(thick + 1), cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # ------------------------------------------------------------------
    # 球
    # ------------------------------------------------------------------
    def _draw_ball(self, img: np.ndarray, hud: HUDState) -> None:
        if hud.current_pos is None:
            return
        x, y = int(round(hud.current_pos[0])), int(round(hud.current_pos[1]))
        color = {
            "yolo": _C_MEASURED,
            "patch": _C_SMOOTHED,
            "pred": _C_PREDICTED,
        }.get(hud.track_method, _C_DIM)
        cv2.circle(img, (x, y), self._px(6), color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), self._px(10), (255, 255, 255), self._px(2), cv2.LINE_AA)
        tag = {"yolo": "MEASURED", "pred": "PREDICTED"}.get(hud.track_method, hud.track_method.upper())
        self._put(img, tag, (x + self._px(14), y - self._px(4)), 0.45, color, 1)

    # ------------------------------------------------------------------
    # 落点
    # ------------------------------------------------------------------
    def _draw_landings(self, img: np.ndarray, hud: HUDState) -> None:
        for lp in hud.landings[-self.landing_history:]:
            color = _C_LANDING_IN if lp.in_table else _C_LANDING_OUT
            cx, cy = int(round(lp.image_px[0])), int(round(lp.image_px[1]))
            cv2.circle(img, (cx, cy), self._px(12), color, self._px(2), cv2.LINE_AA)
            cv2.circle(img, (cx, cy), self._px(3), color, -1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 比分板（左上角）
    # ------------------------------------------------------------------
    def _draw_scoreboard(self, img: np.ndarray, hud: HUDState) -> None:
        pad = self._px(14)
        x0, y0 = self._px(18), self._px(18)
        w, h = self._px(430), self._px(132)
        x1, y1 = x0 + w, y0 + h

        cv2.rectangle(img, (x0, y0), (x1, y1), _C_BG, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), _C_BORDER, self._px(2))

        # 标题
        self._put(img, "AI TABLE TENNIS REFEREE", (x0 + pad, y0 + self._px(26)), 0.52, _C_EVENT, 1)

        # 比分主体：P1  比分  P2
        cy = y0 + self._px(72)
        a, b = hud.score
        self._put(img, "P1", (x0 + pad, cy), 0.72, _C_P1, 2)
        self._put(img, f"{a}", (x0 + self._px(92), cy), 1.35, _C_TEXT, 3)
        self._put(img, ":", (x0 + self._px(148), cy), 1.10, _C_DIM, 2)
        self._put(img, f"{b}", (x0 + self._px(186), cy), 1.35, _C_TEXT, 3)
        self._put(img, "P2", (x0 + self._px(250), cy), 0.72, _C_P2, 2)

        # 发球方指示
        srv_x = x0 + pad + (self._px(6) if hud.server == "A" else self._px(258))
        cv2.circle(img, (srv_x, cy - self._px(9)), self._px(5), _C_SERVER, -1, cv2.LINE_AA)
        self._put(img, "serve", (srv_x + self._px(10), cy - self._px(3)), 0.38, _C_SERVER, 1)

        # 状态行
        state_text = "GAME OVER" if hud.game_over else hud.state
        state_color = (60, 60, 255) if hud.game_over else _C_DIM
        self._put(
            img, f"{state_text}   f{hud.frame_id:04d}   {hud.timestamp_s:6.2f}s",
            (x0 + pad, y0 + self._px(108)), 0.42, state_color, 1,
        )

        # 最近判罚（比分板下方）
        if hud.last_decision and hud.decision_age_s < 3.0:
            by = y1 + self._px(26)
            cv2.rectangle(
                img, (x0, y1 + self._px(8)), (x0 + self._px(560), y1 + self._px(40)), _C_BG, -1
            )
            cv2.rectangle(
                img, (x0, y1 + self._px(8)), (x0 + self._px(560), y1 + self._px(40)),
                _C_BORDER, self._px(1),
            )
            self._put(img, hud.last_decision, (x0 + pad, by), 0.52, _C_EVENT, 1)

    # ------------------------------------------------------------------
    # 2D 俯视雷达（右下角）
    # ------------------------------------------------------------------
    def _draw_radar(self, img: np.ndarray, hud: HUDState) -> None:
        fh, fw = img.shape[:2]
        mw, mh = self._px(240), self._px(134)
        x0, y0 = fw - mw - self._px(30), fh - mh - self._px(30)

        cv2.rectangle(
            img, (x0 - self._px(10), y0 - self._px(28)),
            (x0 + mw + self._px(10), y0 + mh + self._px(10)), _C_BG, -1,
        )
        cv2.rectangle(
            img, (x0 - self._px(10), y0 - self._px(28)),
            (x0 + mw + self._px(10), y0 + mh + self._px(10)), _C_BORDER, self._px(1),
        )
        self._put(
            img, f"TABLE TOP RADAR  {TABLE_LENGTH_CM:.0f}x{TABLE_WIDTH_CM:.1f}cm",
            (x0, y0 - self._px(10)), 0.42, _C_EVENT, 1,
        )

        # 台面
        cv2.rectangle(img, (x0, y0), (x0 + mw, y0 + mh), (95, 60, 25), -1)
        cv2.rectangle(img, (x0, y0), (x0 + mw, y0 + mh), (240, 240, 240), self._px(2))
        cv2.line(img, (x0 + mw // 2, y0), (x0 + mw // 2, y0 + mh), (255, 255, 255), self._px(2))
        self._put(img, "P1", (x0 + self._px(10), y0 + self._px(20)), 0.45, _C_P1, 1)
        self._put(img, "P2", (x0 + mw - self._px(28), y0 + self._px(20)), 0.45, _C_P2, 1)

        # 落点（**只画真实落台事件**，绝不投影空中球 —— 组长 §5）
        for lp in hud.landings[-self.landing_history:]:
            rx = x0 + int(round(lp.x_cm / TABLE_LENGTH_CM * mw))
            ry = y0 + int(round(lp.y_cm / TABLE_WIDTH_CM * mh))
            color = _C_LANDING_IN if lp.in_table else _C_LANDING_OUT
            cv2.circle(img, (rx, ry), self._px(5), color, -1, cv2.LINE_AA)
            cv2.circle(img, (rx, ry), self._px(7), (255, 255, 255), self._px(1), cv2.LINE_AA)

        # 发球方提示
        srv = "P1" if hud.server == "A" else "P2"
        self._put(img, f"serve: {srv}", (x0, y0 + mh + self._px(22)), 0.40, _C_SERVER, 1)
