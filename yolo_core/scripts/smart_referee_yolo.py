"""
Smart Table Tennis Referee System (Clean YOLO11n + Kinematic Multi-Layer Tracker + Physics State Machine)
Deliverable 2: High-recall cross-video ball tracking, 2-frame confirmation, anti-spike correction,
segmented neon trajectory rendering, 2D radar mini-map, and referee engine.
"""

import os
import sys
import time
import argparse
from collections import deque
import cv2
import numpy as np


class ModeCKalmanTracker:
    """
    Pure Kalman Filter specialized for table tennis ball tracking (Mode C).
    - Rapid velocity initialization from 2 confirmed frames
    - Static deadlock rejection (< 2px for 3+ consecutive frames)
    - Short-term prediction interpolation
    """
    def __init__(self, fps=25.0):
        self.fps = fps
        self.dt = 1.0 / fps
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.processNoiseCov[2, 2] = 200.0
        self.kf.processNoiseCov[3, 3] = 200.0
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.prev_coord = None
        self.is_initialized = False
        self.lost_frames = 0
        self.consecutive_tracked = 0
        self.old_raw_pos = None
        self.static_deadlock_count = 0

    def set_fps(self, fps):
        self.fps = fps
        self.dt = 1.0 / fps
        self.kf.transitionMatrix[0, 2] = self.dt
        self.kf.transitionMatrix[1, 3] = self.dt

    def init_with_two_frames(self, pt1, pt2, dt_frames=1):
        dt = self.dt * max(1, dt_frames)
        vx = (pt2[0] - pt1[0]) / dt
        vy = (pt2[1] - pt1[1]) / dt
        self.kf.statePost = np.array([
            [np.float32(pt2[0])],
            [np.float32(pt2[1])],
            [np.float32(vx)],
            [np.float32(vy)]
        ], np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 5.0
        self.prev_coord = pt2
        self.is_initialized = True
        self.lost_frames = 0
        self.consecutive_tracked = 2

    def predict(self):
        if not self.is_initialized:
            return None
        pred = self.kf.predict()
        return (float(pred[0, 0]), float(pred[1, 0]))

    def update(self, coord):
        if not self.is_initialized:
            self.kf.statePost = np.array([
                [np.float32(coord[0])],
                [np.float32(coord[1])],
                [0.0],
                [0.0]
            ], np.float32)
            self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
            self.kf.errorCovPost[2, 2] = 1000.0
            self.kf.errorCovPost[3, 3] = 1000.0
            self.prev_coord = coord
            self.is_initialized = True
            self.lost_frames = 0
            self.consecutive_tracked = 1
            return coord

        if self.prev_coord is not None and (self.kf.statePost[2, 0] == 0.0 and self.kf.statePost[3, 0] == 0.0):
            vx = (coord[0] - self.prev_coord[0]) / self.dt
            vy = (coord[1] - self.prev_coord[1]) / self.dt
            self.kf.statePost[2, 0] = vx
            self.kf.statePost[3, 0] = vy

        measurement = np.array([[np.float32(coord[0])], [np.float32(coord[1])]], np.float32)
        self.kf.correct(measurement)
        self.prev_coord = coord
        self.lost_frames = 0
        self.consecutive_tracked += 1
        return (float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0]))

    def get_velocity(self):
        if not self.is_initialized:
            return (0.0, 0.0)
        return (float(self.kf.statePost[2, 0]), float(self.kf.statePost[3, 0]))

    def reset(self):
        self.is_initialized = False
        self.lost_frames = 0
        self.consecutive_tracked = 0
        self.prev_coord = None
        self.old_raw_pos = None
        self.static_deadlock_count = 0


class SmartRefereeSystem:
    def __init__(self,
                 weights_path="runs/detect/yolo11n_clean/weights/best.pt",
                 table_polygon=None,
                 net_x=929,
                 net_zone=(450, 642),
                 roi_box=(200, 850, 150, 1780),
                 conf_thresh=0.15,
                 imgsz=960):
        self.weights_path = weights_path
        self.roi_box = roi_box
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.net_x = net_x
        self.net_zone = net_zone

        # Load Ultralytics YOLO model
        from ultralytics import YOLO
        if not os.path.exists(weights_path):
            alt_path = "runs/detect/runs/detect/yolo11n_clean/weights/best.pt"
            if os.path.exists(alt_path):
                weights_path = alt_path
        self.model = YOLO(weights_path)
        print(f"[Engine] Loaded YOLO from {weights_path}")

        # Table geometry
        self.table_poly = np.array(table_polygon, dtype=np.int32) if table_polygon is not None else None

        # Homography Setup for 2D Bird's Eye Mini-Map (274cm x 152.5cm)
        self.map_w = 240
        self.map_h = 134
        self.H = None
        if self.table_poly is not None and len(self.table_poly) == 4:
            src = np.array(table_polygon, dtype=np.float32)
            dst = np.array([[0, 0], [self.map_w, 0], [self.map_w, self.map_h], [0, self.map_h]], dtype=np.float32)
            self.H, _ = cv2.findHomography(src, dst)

        # Kinematic Multi-Layer Tracker & Trajectory History
        self.tracker = ModeCKalmanTracker(fps=25.0)
        self.trajectory = deque(maxlen=75)          # (frame_idx, x, y, method)
        self.smooth_vel_history = deque(maxlen=10)   # (vx, vy)
        self.recent_bounces = []                    # [(frame, (bx, by), (mx, my), decision, side)]

        # Kinematic Filters & Anti-Spike State
        self.tentative_pos = None                   # (frame_idx, x, y, conf) for 2-frame confirmation
        self.anti_spike_corrections = 0
        self.long_line_cuts = 0

        # Physics Referee State Machine
        self.score_p1 = 0
        self.score_p2 = 0
        self.last_hit_side = None       # 'LEFT' or 'RIGHT'
        self.last_bounce_side = None    # 'LEFT' or 'RIGHT'
        self.rally_active = False
        self.last_event_text = "READY"
        self.last_event_color = (0, 255, 255)
        self.last_event_frame = -100
        self.last_event_countdown = 0
        self.cooldown_frames = 0        # General event debounce

        # Event statistics counter
        self.stats = {
            'bounce_left': 0,
            'bounce_right': 0,
            'racket_hits': 0,
            'net_hits': 0,
            'out_faults': 0
        }

    def detect_ball_yolo(self, frame, motion_mask=None, pred_pos=None):
        """
        Layer 1: YOLO candidate detection with strict kinematic dual-threshold gating:
        - If tracking:
            <= 45px from pred_pos: conf >= 0.20 (+ motion check if conf < 0.35)
            45px ~ 90px: conf >= 0.38
            > 90px: dropped for tracking
        - If untracked:
            conf >= 0.35 required
        Returns (tracked_cand, global_cand)
        """
        h, w = frame.shape[:2]
        if self.roi_box is not None:
            ymin, ymax, xmin, xmax = self.roi_box
            crop = frame[ymin:ymax, xmin:xmax]
            ox, oy = xmin, ymin
        else:
            crop = frame
            ox, oy = 0, 0

        # Model inference at conf=0.18 to capture candidates down to 0.20
        res = self.model(crop, imgsz=self.imgsz, conf=min(0.18, self.conf_thresh), verbose=False)[0]
        if len(res.boxes) == 0:
            return None, None

        confs = res.boxes.conf.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy()

        tracked_candidate = None
        best_tracked_score = -1e9

        global_candidate = None
        best_global_conf = 0.0

        is_tracking = (self.tracker.is_initialized and pred_pos is not None)

        for b, c in zip(boxes, confs):
            cx = float((b[0] + b[2]) / 2.0 + ox)
            cy = float((b[1] + b[3]) / 2.0 + oy)
            bw = float(b[2] - b[0])
            bh = float(b[3] - b[1])

            # Global candidate for untracked state or 2-frame confirmation of new tracks (conf >= 0.35)
            if c >= 0.35:
                if motion_mask is not None and c < 0.45:
                    bx1, by1 = max(0, int(b[0] + ox)), max(0, int(b[1] + oy))
                    bx2, by2 = min(w, int(b[2] + ox)), min(h, int(b[3] + oy))
                    if np.sum(motion_mask[by1:by2, bx1:bx2]) < 2:
                        continue
                if c > best_global_conf:
                    best_global_conf = float(c)
                    global_candidate = (cx, cy, bw, bh, best_global_conf)

            # If actively tracking, apply strict kinematic dual-threshold gating relative to pred_pos
            if is_tracking:
                dist = np.hypot(cx - pred_pos[0], cy - pred_pos[1])

                if dist <= 45.0:
                    if c < 0.20:
                        continue
                    if c < 0.35 and motion_mask is not None:
                        bx1, by1 = max(0, int(b[0] + ox)), max(0, int(b[1] + oy))
                        bx2, by2 = min(w, int(b[2] + ox)), min(h, int(b[3] + oy))
                        if np.sum(motion_mask[by1:by2, bx1:bx2]) < 2:
                            continue
                elif dist <= 90.0:
                    if c < 0.38:
                        continue
                else:
                    # dist > 90px: strictly dropped for active track
                    continue

                score = c - (dist / 180.0) * 0.1
                if score > best_tracked_score:
                    best_tracked_score = score
                    tracked_candidate = (cx, cy, bw, bh, float(c))

        return tracked_candidate, global_candidate

    def search_local_motion_patch(self, frame_diff, pred_pos, frame_shape):
        """
        Layer 2: Local frame-diff patch search around Kalman prediction with large body motion veto.
        - 50x50 search window around pred_pos
        - Binary diff threshold: absdiff > 20
        - Veto if total motion pixels > 220 (human body/racket motion)
        - Clean motion range: 8 <= total motion pixels <= 120
        - Circularity >= 0.20, distance to pred_pos <= 30px
        """
        if pred_pos is None or frame_diff is None:
            return None

        h, w = frame_shape[:2]
        px, py = pred_pos
        half_w = 25  # 50x50 window
        x1 = max(0, int(px - half_w))
        x2 = min(w, int(px + half_w))
        y1 = max(0, int(py - half_w))
        y2 = min(h, int(py + half_w))

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None

        patch = frame_diff[y1:y2, x1:x2]
        binary = (patch > 20).astype(np.uint8) * 255
        total_motion_pixels = int(np.sum(binary > 0))

        # 人体大运动区一票否决
        if total_motion_pixels > 220:
            return None

        # 背景干净条件: 总运动像素在 8 ~ 120 之间
        if not (8 <= total_motion_pixels <= 120):
            return None

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_pt = None
        min_dist = 999.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 6 <= area <= 100:
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4.0 * np.pi * area / (perimeter * perimeter + 1e-5) if perimeter > 0 else 0.0

                if circularity >= 0.20:
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        cx = float(M["m10"] / M["m00"]) + x1
                        cy = float(M["m01"] / M["m00"]) + y1
                        dist = np.hypot(cx - px, cy - py)
                        if dist <= 30.0 and dist < min_dist:
                            min_dist = dist
                            best_pt = (cx, cy)

        return best_pt

    def apply_anti_spike(self):
        """
        Correction 3: Triplet curvature anti-spike correction.
        For triplet (P_{t-2}, P_{t-1}, P_t):
        If d_spike = ||P_{t-1} - M|| > 28px and d_span = ||P_t - P_{t-2}|| < 90px (where M = (P_{t-2} + P_t)/2),
        replace P_{t-1} with M and increment self.anti_spike_corrections += 1.
        """
        traj = list(self.trajectory)
        n = len(traj)
        if n < 3:
            return

        valid_indices = [i for i, p in enumerate(traj) if p is not None]
        if len(valid_indices) < 3:
            return

        idx_a, idx_b, idx_c = valid_indices[-3], valid_indices[-2], valid_indices[-1]
        pa, pb, pc = traj[idx_a], traj[idx_b], traj[idx_c]

        # Only check if frame gaps are reasonably close (<= 3 frames)
        if (pc[0] - pb[0] <= 3) and (pb[0] - pa[0] <= 3):
            xa, ya = pa[1], pa[2]
            xb, yb = pb[1], pb[2]
            xc, yc = pc[1], pc[2]

            mx = (xa + xc) / 2.0
            my = (ya + yc) / 2.0

            d_spike = float(np.hypot(xb - mx, yb - my))
            d_span = float(np.hypot(xc - xa, yc - ya))

            if d_spike > 28.0 and d_span < 90.0:
                traj[idx_b] = (pb[0], mx, my, "anti_spike")
                self.anti_spike_corrections += 1
                self.trajectory = deque(traj, maxlen=self.trajectory.maxlen)

    def interpolate_trajectory_gaps(self):
        """
        Interpolate small gaps (1~4 frames) in trajectory history for smooth, continuous rendering.
        Strictly limits max interpolation distance to 85px * gap to prevent cross-screen jumps.
        """
        traj = list(self.trajectory)
        n = len(traj)
        if n < 3:
            return

        for i in range(1, n - 1):
            if traj[i] is None:
                # Find preceding valid point
                prev_idx = i - 1
                while prev_idx >= 0 and traj[prev_idx] is None:
                    prev_idx -= 1
                # Find succeeding valid point
                next_idx = i + 1
                while next_idx < n and traj[next_idx] is None:
                    next_idx += 1

                if prev_idx >= 0 and next_idx < n:
                    gap = next_idx - prev_idx
                    if 1 < gap <= 4:
                        p_prev = traj[prev_idx]
                        p_next = traj[next_idx]
                        dist = np.hypot(p_next[1] - p_prev[1], p_next[2] - p_prev[2])
                        if dist <= 85.0 * gap:
                            alpha = (i - prev_idx) / float(gap)
                            interp_x = p_prev[1] + alpha * (p_next[1] - p_prev[1])
                            interp_y = p_prev[2] + alpha * (p_next[2] - p_prev[2])
                            traj[i] = (p_prev[0] + (i - prev_idx), interp_x, interp_y, "interp")

        self.trajectory = deque(traj, maxlen=self.trajectory.maxlen)

    def project_to_minimap(self, pt):
        """Map camera pixel coordinates to mini-map pixel and cm coordinates"""
        if self.H is None:
            return None, None
        p = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        mapped = self.H @ p
        if abs(mapped[2]) < 1e-5:
            return None, None
        mx = float(mapped[0] / mapped[2])
        my = float(mapped[1] / mapped[2])
        cm_x = (mx / self.map_w) * 274.0
        cm_y = (my / self.map_h) * 152.5
        return (mx, my), (cm_x, cm_y)

    def update_physics_engine(self, frame_idx, current_pos, is_pred=False):
        """
        Physics rule-based event detector and referee scoring logic.
        """
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1

        if current_pos is None:
            if self.rally_active and self.tracker.lost_frames > 6:
                if len(self.trajectory) > 0 and self.trajectory[-1] is not None:
                    last_pt = self.trajectory[-1]
                    if last_pt[2] > 700 or last_pt[1] < 120 or last_pt[1] > 1800:
                        self._trigger_out(frame_idx, "Ball Left Play Area")
            return

        cx, cy = current_pos
        vx, vy = self.tracker.get_velocity()
        self.smooth_vel_history.append((vx, vy))

        if len(self.smooth_vel_history) < 3:
            return

        v_curr = self.smooth_vel_history[-1]
        v_prev = self.smooth_vel_history[-2]

        cd_hit = max(5, int(self.tracker.fps * 0.18))
        cd_bounce = max(7, int(self.tracker.fps * 0.22))
        cd_net = max(8, int(self.tracker.fps * 0.25))

        # -----------------------------------------------------------------
        # 1. RACKET HIT: Horizontal Velocity Reversal
        # -----------------------------------------------------------------
        if v_prev[0] * v_curr[0] < 0 and (abs(v_prev[0]) > 2.0 or abs(v_curr[0]) > 2.0) and self.cooldown_frames == 0:
            if v_curr[0] > 0:
                self.last_hit_side = 'LEFT'
                self.rally_active = True
                self.last_bounce_side = None
                self._trigger_event(frame_idx, "RACKET HIT (P1 / Left)", (0, 255, 255), "hit")
                self.stats['racket_hits'] += 1
                self.cooldown_frames = cd_hit
                return
            elif v_curr[0] < 0:
                self.last_hit_side = 'RIGHT'
                self.rally_active = True
                self.last_bounce_side = None
                self._trigger_event(frame_idx, "RACKET HIT (P2 / Right)", (0, 255, 255), "hit")
                self.stats['racket_hits'] += 1
                self.cooldown_frames = cd_hit
                return

        # -----------------------------------------------------------------
        # 2. LEGAL BOUNCE: Vertical Inflection from Downwards to Upwards
        # -----------------------------------------------------------------
        is_vertical_inflection = False
        if v_prev[1] > 1.2 and v_curr[1] < -0.8:
            is_vertical_inflection = True
        elif len(self.trajectory) >= 5:
            y_pts = [p[2] for p in list(self.trajectory)[-5:] if p is not None]
            if len(y_pts) == 5:
                if y_pts[2] >= y_pts[0] + 2 and y_pts[2] >= y_pts[1] and y_pts[2] >= y_pts[3] and y_pts[2] >= y_pts[4] + 2:
                    is_vertical_inflection = True

        if is_vertical_inflection and self.cooldown_frames == 0:
            if v_prev[0] * v_curr[0] >= -1.0:
                dist_to_table = cv2.pointPolygonTest(self.table_poly, (int(cx), int(cy)), True) if self.table_poly is not None else 10.0
                if dist_to_table >= -18.0:
                    bounce_side = 'LEFT' if cx < self.net_x else 'RIGHT'
                    side_label = "LEFT (P1)" if bounce_side == 'LEFT' else "RIGHT (P2)"

                    m_pt, cm_pt = self.project_to_minimap((cx, cy))
                    self.recent_bounces.append({
                        'frame': frame_idx,
                        'img_pos': (int(cx), int(cy)),
                        'map_pos': m_pt,
                        'cm_pos': cm_pt,
                        'decision': f"BOUNCE {side_label}",
                        'is_in': True
                    })

                    # State Machine: Check Double Bounce (连跳违例)
                    if self.rally_active and self.last_bounce_side == bounce_side:
                        if bounce_side == 'LEFT':
                            self.score_p2 += 1
                            self._trigger_event(frame_idx, "DOUBLE BOUNCE on Left! (P2 +1)", (0, 0, 255), "fault")
                        else:
                            self.score_p1 += 1
                            self._trigger_event(frame_idx, "DOUBLE BOUNCE on Right! (P1 +1)", (0, 0, 255), "fault")
                        self.rally_active = False
                        self.last_bounce_side = None
                        self.stats['out_faults'] += 1
                    else:
                        self.last_bounce_side = bounce_side
                        if bounce_side == 'LEFT':
                            self.stats['bounce_left'] += 1
                        else:
                            self.stats['bounce_right'] += 1

                        cm_str = f" [{cm_pt[0]:.0f},{cm_pt[1]:.0f}cm]" if cm_pt else ""
                        self._trigger_event(frame_idx, f"BOUNCE on {side_label}{cm_str}", (0, 255, 0), "bounce")

                    self.cooldown_frames = cd_bounce
                    return
                else:
                    if cy > 670 and self.rally_active:
                        self._trigger_out(frame_idx, "Ball Landed Outside Table")
                        self.cooldown_frames = cd_bounce
                        return

        # -----------------------------------------------------------------
        # 3. NET TOUCH (擦网判定)
        # -----------------------------------------------------------------
        if abs(cx - self.net_x) <= 30 and (self.net_zone[0] <= cy <= self.net_zone[1]) and self.cooldown_frames == 0:
            if abs(v_curr[0]) < 0.5 * max(abs(v_prev[0]), 5.0):
                self._trigger_event(frame_idx, "NET TOUCH / CORD HIT", (0, 165, 255), "net")
                self.stats['net_hits'] += 1
                self.cooldown_frames = cd_net

    def _trigger_out(self, frame_idx, reason):
        if not self.rally_active:
            return
        if self.last_hit_side == 'LEFT':
            self.score_p2 += 1
            self._trigger_event(frame_idx, f"FAULT (P1 Out: {reason}) -> P2 +1", (0, 0, 255), "fault")
        elif self.last_hit_side == 'RIGHT':
            self.score_p1 += 1
            self._trigger_event(frame_idx, f"FAULT (P2 Out: {reason}) -> P1 +1", (0, 0, 255), "fault")
        self.rally_active = False
        self.last_bounce_side = None
        self.stats['out_faults'] += 1
        self.cooldown_frames = max(8, int(self.tracker.fps * 0.25))

    def _trigger_event(self, frame_idx, text, color, ev_type):
        self.last_event_text = text
        self.last_event_color = color
        self.last_event_frame = frame_idx
        self.last_event_countdown = max(28, int(self.tracker.fps * 1.2))
        print(f"[{frame_idx:04d}] >>> REFEREE EVENT: {text}")

    def render_hud(self, vis, frame_idx, current_pos, track_method="yolo"):
        h, w = vis.shape[:2]

        # 1. Draw Table Geometry and Half-Table Boundary
        if self.table_poly is not None:
            cv2.polylines(vis, [self.table_poly], True, (255, 220, 0), 2)
            cv2.line(vis, (self.net_x, int(self.table_poly[0][1])), 
                          (self.net_x, int(self.table_poly[2][1])), (255, 0, 255), 2)

            # Half-Court Labels
            p1_label_pos = (self.table_poly[3][0] + 40, min(h - 20, self.table_poly[3][1] - 15))
            p2_label_pos = (min(w - 180, self.net_x + 50), min(h - 20, self.table_poly[2][1] - 15))
            cv2.putText(vis, "PLAYER 1 (LEFT)", p1_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 220, 0), 2)
            cv2.putText(vis, "PLAYER 2 (RIGHT)", p2_label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 220, 0), 2)

        # 2. Draw Continuous Trajectory Line (Glowing Neon Trail with Segmented Cuts & 3-pt Smoothing)
        raw_pts = [p for p in list(self.trajectory) if p is not None]
        if len(raw_pts) >= 2:
            segments = []
            curr_seg = [raw_pts[0]]
            for i in range(1, len(raw_pts)):
                p_prev = curr_seg[-1]
                p_curr = raw_pts[i]
                df = p_curr[0] - p_prev[0]
                dd = np.hypot(p_curr[1] - p_prev[1], p_curr[2] - p_prev[2])
                if df > 2 or dd > 85.0:
                    segments.append(curr_seg)
                    curr_seg = [p_curr]
                else:
                    curr_seg.append(p_curr)
            if len(curr_seg) > 0:
                segments.append(curr_seg)

            for seg in segments:
                m = len(seg)
                if m < 2:
                    continue

                # Apply 3-point weighted smoothing: 0.25*P_{i-1} + 0.50*P_i + 0.25*P_{i+1}
                smoothed_coords = []
                for j in range(m):
                    if j == 0 or j == m - 1:
                        smoothed_coords.append((seg[j][1], seg[j][2], seg[j][0]))
                    else:
                        sx = 0.25 * seg[j - 1][1] + 0.50 * seg[j][1] + 0.25 * seg[j + 1][1]
                        sy = 0.25 * seg[j - 1][2] + 0.50 * seg[j][2] + 0.25 * seg[j + 1][2]
                        smoothed_coords.append((sx, sy, seg[j][0]))

                # Render smooth lines strictly within the segment
                for k in range(1, len(smoothed_coords)):
                    p0 = smoothed_coords[k - 1]
                    p1 = smoothed_coords[k]
                    seg_dist = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
                    if seg_dist > 85.0:
                        continue  # Safety guard: never draw across > 85px

                    # Glowing gradient based on point age
                    age_norm = max(0.0, min(1.0, (p1[2] - (frame_idx - self.trajectory.maxlen)) / float(self.trajectory.maxlen)))
                    color_core = (int(40 * (1 - age_norm)), int(255 * age_norm), int(255 * (1 - age_norm * 0.4)))
                    color_glow = (int(color_core[0] * 0.4), int(color_core[1] * 0.4), int(color_core[2] * 0.4))

                    pt0_i = (int(p0[0]), int(p0[1]))
                    pt1_i = (int(p1[0]), int(p1[1]))

                    # Outer glow layer
                    cv2.line(vis, pt0_i, pt1_i, color_glow, 6, cv2.LINE_AA)
                    # Inner core neon ribbon
                    cv2.line(vis, pt0_i, pt1_i, color_core, 2, cv2.LINE_AA)

        # 3. Draw Ball Detection / Tracking Dot
        if current_pos is not None:
            bx, by = int(current_pos[0]), int(current_pos[1])
            if track_method == "yolo":
                dot_color = (0, 255, 0)
                status_tag = "YOLO11n"
            elif track_method == "patch":
                dot_color = (0, 255, 255)
                status_tag = "MOTION PATCH"
            elif track_method == "interp":
                dot_color = (255, 100, 255)
                status_tag = "INTERPOLATED"
            elif track_method == "anti_spike":
                dot_color = (0, 200, 255)
                status_tag = "SMOOTHED"
            else:
                dot_color = (0, 165, 255)
                status_tag = "KF PRED"

            cv2.circle(vis, (bx, by), 6, dot_color, -1)
            cv2.circle(vis, (bx, by), 10, (255, 255, 255), 2)
            cv2.putText(vis, status_tag, (bx + 14, by - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, dot_color, 2)

        # 4. Draw Recent Bounce Markers on Table
        for b in self.recent_bounces[-5:]:
            pos = b['img_pos']
            color = (0, 255, 0) if b['is_in'] else (0, 0, 255)
            cv2.circle(vis, pos, 12, color, 2)
            cv2.circle(vis, pos, 3, color, -1)

        # 5. Top Broadcast Scoreboard HUD
        hud_w = min(680, w - 50)
        cv2.rectangle(vis, (25, 20), (25 + hud_w, 140), (20, 20, 20), -1)
        cv2.rectangle(vis, (25, 20), (25 + hud_w, 140), (100, 100, 100), 2)

        cv2.putText(vis, "AI TABLE TENNIS SMART REFEREE (YOLO11n + KF Mode C)", 
                    (40, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)

        score_text = f"SCORE:  [ Player 1: {self.score_p1} ]   vs   [ Player 2: {self.score_p2} ]"
        cv2.putText(vis, score_text, (40, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        rally_state = "RALLY IN PROGRESS" if self.rally_active else "READY / DEAD BALL"
        rally_color = (0, 255, 0) if self.rally_active else (180, 180, 180)
        status_line = f"Status: {rally_state} | Frm {frame_idx:04d} | Spikes Fixed: {self.anti_spike_corrections} | Cuts: {self.long_line_cuts}"
        cv2.putText(vis, status_line, (40, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.44, rally_color, 1)

        # Event Banner
        if self.last_event_countdown > 0:
            self.last_event_countdown -= 1
            cv2.putText(vis, f"DECISION: {self.last_event_text}", 
                        (40, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.58, self.last_event_color, 2)
        else:
            cv2.putText(vis, f"DECISION: {self.last_event_text}", 
                        (40, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 150, 150), 1)

        # 6. Mini-Map 2D Bird's-Eye View (Bottom-Right Corner)
        if self.H is not None:
            mw, mh = self.map_w, self.map_h
            map_x0 = w - mw - 30
            map_y0 = h - mh - 30

            cv2.rectangle(vis, (map_x0 - 10, map_y0 - 28), (map_x0 + mw + 10, map_y0 + mh + 10), (15, 15, 15), -1)
            cv2.rectangle(vis, (map_x0 - 10, map_y0 - 28), (map_x0 + mw + 10, map_y0 + mh + 10), (120, 120, 120), 1)
            cv2.putText(vis, "2D TABLE TOP RADAR (274x152.5 cm)", (map_x0, map_y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

            cv2.rectangle(vis, (map_x0, map_y0), (map_x0 + mw, map_y0 + mh), (130, 60, 20), -1)
            cv2.rectangle(vis, (map_x0, map_y0), (map_x0 + mw, map_y0 + mh), (255, 255, 255), 2)
            cv2.line(vis, (map_x0 + mw // 2, map_y0), (map_x0 + mw // 2, map_y0 + mh), (255, 255, 255), 2)
            cv2.line(vis, (map_x0, map_y0 + mh // 2), (map_x0 + mw, map_y0 + mh // 2), (200, 200, 200), 1)

            cv2.putText(vis, "P1", (map_x0 + 10, map_y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(vis, "P2", (map_x0 + mw - 25, map_y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            for b in self.recent_bounces[-5:]:
                if b['map_pos'] is not None:
                    mx, my = b['map_pos']
                    if 0 <= mx <= mw and 0 <= my <= mh:
                        rx = int(map_x0 + mx)
                        ry = int(map_y0 + my)
                        color = (0, 255, 0) if b['is_in'] else (0, 0, 255)
                        cv2.circle(vis, (rx, ry), 5, color, -1)
                        cv2.circle(vis, (rx, ry), 7, (255, 255, 255), 1)

            if current_pos is not None:
                m_curr, _ = self.project_to_minimap(current_pos)
                if m_curr is not None:
                    mx, my = m_curr
                    if -20 <= mx <= mw + 20 and -20 <= my <= mh + 20:
                        rx = int(map_x0 + np.clip(mx, 0, mw))
                        ry = int(map_y0 + np.clip(my, 0, mh))
                        cv2.circle(vis, (rx, ry), 4, (0, 255, 255), -1)

    def process_video(self, video_path, output_path=None, max_frames=None):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.tracker.set_fps(fps)
        self.trajectory = deque(maxlen=max(45, int(fps * 1.5)))

        frames_to_run = min(total_frames, max_frames) if max_frames else total_frames
        print(f"\n[Start] Processing {video_path} ({w}x{h} @ {fps}fps, {frames_to_run} frames)...")

        writer = None
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        frame_idx = 0
        t_start = time.time()
        prev_gray = None
        tracked_frames_count = 0

        while cap.isOpened() and frame_idx < frames_to_run:
            ret, frame = cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(prev_gray, curr_gray) if prev_gray is not None else None
            motion_mask = (frame_diff > 15).astype(np.uint8) if frame_diff is not None else None

            # 1. Kalman prediction
            pred_pos = self.tracker.predict()

            # 2. Layer 1: YOLO Detection + Kinematic Dual-Threshold Filter
            tracked_cand, global_cand = self.detect_ball_yolo(frame, motion_mask=motion_mask, pred_pos=pred_pos)

            final_pos = None
            track_method = "none"

            # 3. Tracking & 2-Frame Confirmation State Machine
            if self.tracker.is_initialized:
                # Check static deadlock on tracked candidate
                is_static_deadlock = False
                if tracked_cand is not None:
                    cx, cy = tracked_cand[0], tracked_cand[1]
                    if self.tracker.old_raw_pos is not None:
                        d_static = np.hypot(cx - self.tracker.old_raw_pos[0], cy - self.tracker.old_raw_pos[1])
                        if d_static < 2.0:
                            self.tracker.static_deadlock_count += 1
                            if self.tracker.static_deadlock_count >= max(3, int(fps * 0.08)):
                                is_static_deadlock = True
                        else:
                            self.tracker.static_deadlock_count = 0
                    else:
                        self.tracker.static_deadlock_count = 0
                    self.tracker.old_raw_pos = (cx, cy)

                if tracked_cand is not None and not is_static_deadlock:
                    final_pos = (tracked_cand[0], tracked_cand[1])
                    self.tracker.update(final_pos)
                    track_method = "yolo"
                    self.tentative_pos = None
                else:
                    if is_static_deadlock:
                        self.tracker.reset()

                    # Layer 1 missed or deadlocked -> Layer 2: Local frame-diff motion patch
                    patch_pos = None
                    if pred_pos is not None and frame_diff is not None and self.tracker.lost_frames <= 3:
                        patch_pos = self.search_local_motion_patch(frame_diff, pred_pos, (h, w))

                    if patch_pos is not None:
                        final_pos = patch_pos
                        self.tracker.update(final_pos)
                        track_method = "patch"
                        self.tentative_pos = None
                    else:
                        # Layer 2 missed -> Short-term pure Kalman extrapolation
                        max_extrapolate = max(2, int(fps * 0.08))
                        if self.tracker.consecutive_tracked >= 2 and self.tracker.lost_frames < max_extrapolate and pred_pos is not None:
                            final_pos = pred_pos
                            track_method = "pred"
                            self.tracker.lost_frames += 1
                        else:
                            final_pos = None
                            self.tracker.lost_frames += 1
                            if self.tracker.lost_frames > max(4, int(fps * 0.15)):
                                self.tracker.reset()

                        # If tracking is broken and a distant ball candidate exists (> 90px),
                        # begin 2-Frame Confirmation for new rally re-acquisition (require lost_frames >= 3)
                        if self.tracker.lost_frames >= 3 and global_cand is not None and (pred_pos is None or np.hypot(global_cand[0] - pred_pos[0], global_cand[1] - pred_pos[1]) > 90.0):
                            if self.tentative_pos is None:
                                self.tentative_pos = (frame_idx, global_cand[0], global_cand[1], global_cand[4])
                            else:
                                t_f, t_x, t_y, _ = self.tentative_pos
                                if frame_idx == t_f + 1:
                                    dist_tent = np.hypot(global_cand[0] - t_x, global_cand[1] - t_y)
                                    if 3.5 <= dist_tent <= 65.0:
                                        self.tracker.reset()
                                        self.tracker.init_with_two_frames((t_x, t_y), (global_cand[0], global_cand[1]))
                                        if len(self.trajectory) > 0 and self.trajectory[-1] is None:
                                            self.trajectory[-1] = (t_f, t_x, t_y, "yolo")
                                            tracked_frames_count += 1
                                        final_pos = (global_cand[0], global_cand[1])
                                        track_method = "yolo"
                                        self.tentative_pos = None
                                    else:
                                        self.tentative_pos = (frame_idx, global_cand[0], global_cand[1], global_cand[4]) if dist_tent >= 3.5 else None
                                else:
                                    self.tentative_pos = (frame_idx, global_cand[0], global_cand[1], global_cand[4])

            else:
                # Untracked State: Strict 2-Frame Confirmation (with anti-static threshold >= 3.5px)
                if global_cand is not None:
                    if self.tentative_pos is None:
                        self.tentative_pos = (frame_idx, global_cand[0], global_cand[1], global_cand[4])
                        final_pos = None
                    else:
                        t_f, t_x, t_y, _ = self.tentative_pos
                        if frame_idx == t_f + 1:
                            dist_tent = np.hypot(global_cand[0] - t_x, global_cand[1] - t_y)
                            if 3.5 <= dist_tent <= 65.0:
                                # Confirmed over 2 consecutive frames moving like a real ball!
                                self.tracker.init_with_two_frames((t_x, t_y), (global_cand[0], global_cand[1]))
                                if len(self.trajectory) > 0 and self.trajectory[-1] is None:
                                    self.trajectory[-1] = (t_f, t_x, t_y, "yolo")
                                    tracked_frames_count += 1
                                final_pos = (global_cand[0], global_cand[1])
                                track_method = "yolo"
                                self.tentative_pos = None
                            else:
                                self.tentative_pos = (frame_idx, global_cand[0], global_cand[1], global_cand[4]) if dist_tent >= 3.5 else None
                                final_pos = None
                        else:
                            self.tentative_pos = (frame_idx, global_cand[0], global_cand[1], global_cand[4])
                            final_pos = None
                else:
                    self.tentative_pos = None
                    final_pos = None

            # 4. Trajectory Recording, Jump Cutting, Anti-Spike & Gap Interpolation
            if final_pos is not None:
                # Check for cross-screen jump cut relative to preceding valid point
                prev_valid = None
                for p in reversed(self.trajectory):
                    if p is not None:
                        prev_valid = p
                        break
                if prev_valid is not None:
                    jump_dist = np.hypot(final_pos[0] - prev_valid[1], final_pos[1] - prev_valid[2])
                    if jump_dist > 85.0:
                        self.long_line_cuts += 1

                self.trajectory.append((frame_idx, final_pos[0], final_pos[1], track_method))
                tracked_frames_count += 1
                self.apply_anti_spike()
                self.interpolate_trajectory_gaps()
            else:
                self.trajectory.append(None)

            # 5. Physics Engine Event Detection & Decision State Machine
            self.update_physics_engine(frame_idx, final_pos, is_pred=(track_method in ("pred", "interp")))

            # 6. Render HUD and Mini-Map
            if writer is not None:
                vis = frame.copy()
                self.render_hud(vis, frame_idx, final_pos, track_method=track_method)
                writer.write(vis)

            prev_gray = curr_gray
            frame_idx += 1
            if frame_idx % 150 == 0:
                print(f"[Run] Processed {frame_idx}/{frames_to_run} frames ({frame_idx/(time.time() - t_start):.1f} FPS)...")

        cap.release()
        if writer is not None:
            writer.release()

        elapsed = time.time() - t_start
        coverage_pct = (tracked_frames_count / max(1, frame_idx)) * 100.0

        print(f"\n" + "=" * 65)
        print(f"PROCESSING COMPLETED: {video_path}")
        print(f"Time: {elapsed:.2f}s | Speed: {frame_idx/elapsed:.1f} FPS")
        print(f"Trajectory Coverage: {tracked_frames_count}/{frame_idx} frames ({coverage_pct:.1f}%)")
        print(f"Kinematic Filters: Anti-Spike Corrections: {self.anti_spike_corrections} | Cross-Screen Lines Cut: {self.long_line_cuts}")
        print(f"Final Score: Player 1 [{self.score_p1}] vs [{self.score_p2}] Player 2")
        print(f"Events Summary: Left Bounces: {self.stats['bounce_left']}, Right Bounces: {self.stats['bounce_right']}, Racket Hits: {self.stats['racket_hits']}, Net Hits: {self.stats['net_hits']}, Faults: {self.stats['out_faults']}")
        if output_path:
            print(f"Saved Output Video: {output_path}")
        print("=" * 65 + "\n")

        return {
            'video': video_path,
            'frames': frame_idx,
            'tracked_frames': tracked_frames_count,
            'coverage_pct': round(coverage_pct, 1),
            'fps': round(frame_idx / elapsed, 1),
            'anti_spike_corrections': self.anti_spike_corrections,
            'long_line_cuts': self.long_line_cuts,
            'score_p1': self.score_p1,
            'score_p2': self.score_p2,
            'stats': self.stats,
            'output_path': output_path
        }


def main():
    parser = argparse.ArgumentParser(description="Smart Table Tennis Referee System (YOLO11n Clean + Kinematic Tracker)")
    parser.add_argument('--video', type=str, default="table-tennis-detection/videos/test_game_2.avi", help="Video path")
    parser.add_argument('--weights', type=str, default="runs/detect/yolo11n_clean/weights/best.pt", help="Weights path")
    parser.add_argument('--output', type=str, default=None, help="Output video path")
    parser.add_argument('--conf', type=float, default=None, help="Confidence threshold")
    parser.add_argument('--imgsz', type=int, default=960, help="Inference image size")
    parser.add_argument('--max-frames', type=int, default=None, help="Max frames to process")
    args = parser.parse_args()

    # Detect resolution and calibrate preset
    cap_probe = cv2.VideoCapture(args.video)
    vw = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    vh = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cap_probe.release()

    video_lower = args.video.lower()

    if "rally" in video_lower:
        # Calibrated precisely to test_rally.mp4 broadcast perspective
        base_w, base_h = 1920.0, 1080.0
        raw_poly = [[587, 575], [1305, 450], [1630, 565], [765, 786]]
        raw_roi = [350, 860, 80, 1850]
        raw_net_x = 1148
        raw_net_zone = (450, 680)
        default_out = "output_final_rally_yolo11n.mp4"
    elif "plos" in video_lower:
        # Calibrated to test_plos.mp4 throwing experiment (1024x576)
        base_w, base_h = 1024.0, 576.0
        raw_poly = [[750, 440], [1010, 440], [1010, 560], [750, 560]]
        raw_roi = [120, 560, 200, 1024]
        raw_net_x = 880
        raw_net_zone = (350, 560)
        default_out = "output_final_plos_yolo11n.mp4"
    else:
        # Calibrated to OpenTTGames Game 2 / Test 2 (1080p standard view)
        base_w, base_h = 1920.0, 1080.0
        raw_poly = [[557, 519], [1329, 535], [1473, 654], [367, 631]]
        raw_roi = [200, 850, 150, 1780]
        raw_net_x = 929
        raw_net_zone = (450, 642)
        default_out = "output_final_game2_yolo11n.mp4"

    # Scale geometry to actual video resolution
    sx = vw / base_w
    sy = vh / base_h
    table_poly = [[int(p[0] * sx), int(p[1] * sy)] for p in raw_poly]
    roi_box = (int(raw_roi[0] * sy), int(raw_roi[1] * sy), int(raw_roi[2] * sx), int(raw_roi[3] * sx))
    net_x = int(raw_net_x * sx)
    net_zone = (int(raw_net_zone[0] * sy), int(raw_net_zone[1] * sy))

    conf_thresh = args.conf if args.conf is not None else 0.15
    out_video = args.output if args.output else default_out

    referee = SmartRefereeSystem(
        weights_path=args.weights,
        table_polygon=table_poly,
        net_x=net_x,
        net_zone=net_zone,
        roi_box=roi_box,
        conf_thresh=conf_thresh,
        imgsz=args.imgsz
    )

    res = referee.process_video(args.video, output_path=out_video, max_frames=args.max_frames)
    return res


if __name__ == '__main__':
    main()
