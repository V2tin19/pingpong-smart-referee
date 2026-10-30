"""
Evaluate Pipeline for Table Tennis Ball Detection
Benchmark runner for Mode A (Pure YOLO + ROI Crop) and Mode B (ROI + Motion Filter + Kalman Recovery).
Supports both ONNX models (YOLOv5) and PyTorch models (Ultralytics YOLOv8 / YOLO11).
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import pandas as pd

class KalmanBallTracker:
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
        return (float(self.kf.statePost[0, 0]), float(self.kf.statePost[1, 0]))

    def reset(self):
        self.is_initialized = False
        self.lost_frames = 0
        self.prev_coord = None


class BallDetectorModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.is_onnx = model_path.lower().endswith('.onnx')
        if self.is_onnx:
            self.net = cv2.dnn.readNet(model_path)
            # Try to use CUDA if available in OpenCV
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except Exception:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self.output_layers = self.net.getUnconnectedOutLayersNames()
            self.model = None
        else:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.net = None

    def detect(self, crop_img, conf_thresh=0.15):
        h, w = crop_img.shape[:2]
        candidates = []  # list of (cx, cy, bw, bh, conf) in crop coords

        if self.is_onnx:
            blob = cv2.dnn.blobFromImage(crop_img, 1.0 / 255.0, (640, 640), [0, 0, 0], 1, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)
            rows = outputs[0].shape[1]
            x_factor = w / 640.0
            y_factor = h / 640.0

            for r in range(rows):
                row = outputs[0][0][r]
                obj_conf = row[4]
                if obj_conf >= conf_thresh:
                    class_scores = row[5:]
                    class_id = int(np.argmax(class_scores))
                    class_score = class_scores[class_id]
                    total_conf = float(obj_conf * class_score)
                    if total_conf >= conf_thresh:
                        cx = float(row[0] * x_factor)
                        cy = float(row[1] * y_factor)
                        bw = float(row[2] * x_factor)
                        bh = float(row[3] * y_factor)
                        candidates.append((cx, cy, bw, bh, total_conf))
        else:
            res = self.model(crop_img, conf=conf_thresh, verbose=False)[0]
            if len(res.boxes) > 0:
                confs = res.boxes.conf.cpu().numpy()
                boxes = res.boxes.xyxy.cpu().numpy()
                for b, c in zip(boxes, confs):
                    cx = float((b[0] + b[2]) / 2.0)
                    cy = float((b[1] + b[3]) / 2.0)
                    bw = float(b[2] - b[0])
                    bh = float(b[3] - b[1])
                    candidates.append((cx, cy, bw, bh, float(c)))

        return candidates


def evaluate_video(model_path,
                   video_path="table-tennis-detection/videos/test_game_2.avi",
                   gt_csv_path="table-tennis-detection/data_annotations/test_dataset_info.csv",
                   mode='A',
                   roi_box=[200, 850, 150, 1780],
                   displacement_limit=180.0,
                   conf_thresh=0.15,
                   output_video=None):
    """
    Run evaluation on the test video.
    Returns metrics dict for All frames (0~778) and Test split (390~778).
    """
    ymin, ymax, xmin, xmax = roi_box
    offset_x, offset_y = xmin, ymin
    crop_w = xmax - xmin
    crop_h = ymax - ymin

    # Load Ground Truth
    df = pd.read_csv(gt_csv_path)
    df_game = df[df['game_id'] == 2].sort_values('frame_id').reset_index(drop=True)
    total_frames = len(df_game)

    cap = cv2.VideoCapture(video_path)
    actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = BallDetectorModel(model_path)
    tracker = KalmanBallTracker(fps=fps)
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    writer = None
    if output_video:
        os.makedirs(os.path.dirname(os.path.abspath(output_video)), exist_ok=True)
        writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_idx = 0
    detections_log = []
    old_pos = None
    old_raw_pos = None
    static_deadlock_count = 0
    consecutive_tracked = 0

    t_infer_total = 0.0
    infer_count = 0

    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. ROI crop
        crop = frame[ymin:ymax, xmin:xmax]

        # 2. MOG2 Motion mask (only computed if Mode B)
        fgMask = None
        if mode == 'B':
            blurred = cv2.GaussianBlur(crop, (9, 9), 2)
            fgMask = backSub.apply(blurred)

        # 3. Kalman prediction
        pred_pos = tracker.predict()

        # 4. Model inference
        t0 = time.time()
        raw_candidates = detector.detect(crop, conf_thresh=conf_thresh)
        t_infer = time.time() - t0
        t_infer_total += t_infer
        infer_count += 1

        chosen_ball = None  # (cx, cy, bw, bh, conf) in full coords

        # Candidate selection with displacement limit (180px)
        valid_candidates = []
        for cand in raw_candidates:
            cx_full = cand[0] + offset_x
            cy_full = cand[1] + offset_y
            if old_pos is not None:
                dist = np.sqrt((cx_full - old_pos[0])**2 + (cy_full - old_pos[1])**2)
                if dist <= displacement_limit:
                    valid_candidates.append((cx_full, cy_full, cand[2], cand[3], cand[4]))
                elif cand[4] >= 0.50:  # High confidence re-acquisition
                    valid_candidates.append((cx_full, cy_full, cand[2], cand[3], cand[4]))
            else:
                valid_candidates.append((cx_full, cy_full, cand[2], cand[3], cand[4]))

        # Sort by confidence descending
        if valid_candidates:
            valid_candidates.sort(key=lambda x: x[4], reverse=True)
            chosen_ball = valid_candidates[0]

        final_pos = None
        method_used = "missed"

        if mode == 'A':
            # Mode A: Pure YOLO + ROI crop
            if chosen_ball is not None:
                final_pos = (chosen_ball[0], chosen_ball[1])
                method_used = "yolo"
                old_pos = final_pos
            else:
                final_pos = None
        elif mode == 'B':
            # Mode B: ROI + Motion Filter + Kalman Short-term Compensation
            is_valid_motion = False
            if chosen_ball is not None:
                bx1 = max(0, int(chosen_ball[0] - offset_x - chosen_ball[2] / 2.0 - 2))
                by1 = max(0, int(chosen_ball[1] - offset_y - chosen_ball[3] / 2.0 - 2))
                bx2 = min(crop_w, int(chosen_ball[0] - offset_x + chosen_ball[2] / 2.0 + 2))
                by2 = min(crop_h, int(chosen_ball[1] - offset_y + chosen_ball[3] / 2.0 + 2))
                patch = fgMask[by1:by2, bx1:bx2]
                motion_pixels = np.sum(patch > 0)
                # If motion pixels > 0, not completely static
                if motion_pixels >= 1 or chosen_ball[4] >= 0.40:
                    is_valid_motion = True

            if chosen_ball is not None and is_valid_motion:
                tracked = tracker.update((chosen_ball[0], chosen_ball[1]))
                final_pos = tracked
                method_used = "yolo"
                old_pos = final_pos
            else:
                # YOLO missed or static false alarm -> Try short-term Kalman recovery
                if tracker.is_initialized and tracker.lost_frames < 3 and pred_pos is not None:
                    # Look for active motion blob around pred_pos in 80x80 window
                    px_crop = pred_pos[0] - offset_x
                    py_crop = pred_pos[1] - offset_y
                    x1_win = max(0, int(px_crop - 40))
                    y1_win = max(0, int(py_crop - 40))
                    x2_win = min(crop_w, int(px_crop + 40))
                    y2_win = min(crop_h, int(py_crop + 40))

                    search_patch = fgMask[y1_win:y2_win, x1_win:x2_win]
                    contours, _ = cv2.findContours(search_patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    best_blob = None
                    best_blob_dist = float('inf')

                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if 8 <= area <= 380:
                            M = cv2.moments(cnt)
                            if M["m00"] > 0:
                                bcx = M["m10"] / M["m00"] + x1_win
                                bcy = M["m01"] / M["m00"] + y1_win
                                d = np.sqrt((bcx - px_crop)**2 + (bcy - py_crop)**2)
                                if d < best_blob_dist:
                                    best_blob_dist = d
                                    best_blob = (bcx + offset_x, bcy + offset_y)

                    if best_blob is not None:
                        final_pos = tracker.update(best_blob)
                        method_used = "kalman_mog2"
                        old_pos = final_pos
                    elif tracker.lost_frames < 2:
                        # 1 frame pure Kalman extrapolation
                        final_pos = pred_pos
                        method_used = "kalman_pred"
                        old_pos = final_pos
                        tracker.lost_frames += 1
                    else:
                        tracker.lost_frames += 1
                        old_pos = None
                else:
                    tracker.lost_frames += 1
                    if tracker.lost_frames > 5:
                        tracker.reset()
                    old_pos = None

        elif mode == 'C':
            # Mode C: Lightweight Tracking Aid (Pure Kalman Smooth + Static Deadlock Prevention)
            is_static_deadlock = False
            if chosen_ball is not None:
                if old_raw_pos is not None:
                    d_static = np.sqrt((chosen_ball[0] - old_raw_pos[0])**2 + (chosen_ball[1] - old_raw_pos[1])**2)
                    if d_static < 2.0:
                        static_deadlock_count += 1
                        if static_deadlock_count >= 3:
                            is_static_deadlock = True
                    else:
                        static_deadlock_count = 0
                else:
                    static_deadlock_count = 0
                old_raw_pos = (chosen_ball[0], chosen_ball[1])

            if chosen_ball is not None and not is_static_deadlock:
                final_pos = (chosen_ball[0], chosen_ball[1])
                tracker.update(final_pos)
                consecutive_tracked += 1
                tracker.lost_frames = 0
                method_used = "yolo"
                old_pos = final_pos
            else:
                # YOLO missed or static deadlock -> Short-term pure Kalman extrapolation
                if tracker.is_initialized and consecutive_tracked >= 2 and tracker.lost_frames < 3 and pred_pos is not None:
                    final_pos = pred_pos
                    method_used = "kalman_pred"
                    old_pos = final_pos
                    tracker.lost_frames += 1
                else:
                    final_pos = None
                    tracker.lost_frames += 1
                    if tracker.lost_frames > 5:
                        tracker.reset()
                        consecutive_tracked = 0
                    old_pos = None


        # Compare with Ground Truth
        gt_row = df_game.iloc[frame_idx]
        gt_has_ball = False
        gt_cx, gt_cy = None, None
        dist_err = None

        if pd.notna(gt_row['a']) and pd.notna(gt_row['b']):
            gt_has_ball = True
            gt_cx = float(gt_row['a'] * 1920.0)
            gt_cy = float(gt_row['b'] * 1080.0)

        if final_pos is not None:
            if gt_has_ball:
                dist_err = float(np.sqrt((final_pos[0] - gt_cx)**2 + (final_pos[1] - gt_cy)**2))
            else:
                dist_err = float('inf')  # False positive

        detections_log.append({
            'frame_idx': frame_idx,
            'gt_has_ball': gt_has_ball,
            'has_pred': final_pos is not None,
            'gt_center': (gt_cx, gt_cy) if gt_has_ball else None,
            'pred_center': final_pos,
            'dist_error': dist_err,
            'method': method_used
        })

        if writer is not None:
            vis = frame.copy()
            cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (0, 165, 255), 1)
            if gt_has_ball:
                cv2.circle(vis, (int(gt_cx), int(gt_cy)), 10, (255, 255, 0), 2)
            if final_pos is not None:
                color = (0, 255, 0) if method_used == "yolo" else (0, 165, 255)
                cv2.circle(vis, (int(final_pos[0]), int(final_pos[1])), 6, color, -1)
                lbl = f"{method_used.upper()}"
                if dist_err is not None and dist_err != float('inf'):
                    lbl += f" {dist_err:.1f}px"
                cv2.putText(vis, lbl, (int(final_pos[0]) + 10, int(final_pos[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            writer.write(vis)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    fps_pure_infer = infer_count / t_infer_total if t_infer_total > 0 else 0.0

    def calculate_metrics(log_slice, name):
        total_f = len(log_slice)
        gt_balls = sum(1 for x in log_slice if x['gt_has_ball'])

        def get_prf1(tol):
            tp = sum(1 for x in log_slice if x['gt_has_ball'] and x['dist_error'] is not None and x['dist_error'] <= tol)
            fp = sum(1 for x in log_slice if x['has_pred'] and (not x['gt_has_ball'] or x['dist_error'] is None or x['dist_error'] > tol))
            fn = sum(1 for x in log_slice if x['gt_has_ball'] and (not x['has_pred'] or x['dist_error'] > tol))
            precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
            recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            return tp, fp, fn, precision, recall, f1

        tp15, fp15, fn15, p15, r15, f15 = get_prf1(15)
        tp25, fp25, fn25, p25, r25, f25 = get_prf1(25)

        valid_errs = [x['dist_error'] for x in log_slice if x['dist_error'] is not None and x['dist_error'] != float('inf')]
        mae = float(np.mean(valid_errs)) if valid_errs else float('nan')

        return {
            'scope': name,
            'total_frames': total_f,
            'gt_ball_frames': gt_balls,
            'recall_15px': round(r15, 2),
            'recall_25px': round(r25, 2),
            'precision_15px': round(p15, 2),
            'precision_25px': round(p25, 2),
            'f1_15px': round(f15, 2),
            'f1_25px': round(f25, 2),
            'mae_px': round(mae, 2),
            'tp25': tp25,
            'fp25': fp25,
            'fn25': fn25
        }

    metrics_all = calculate_metrics(detections_log, "Full (0-778)")
    metrics_test = calculate_metrics(detections_log[390:], "Test Split (390-778)")

    return {
        'model': os.path.basename(model_path),
        'mode': mode,
        'fps_infer': round(fps_pure_infer, 1),
        'full': metrics_all,
        'test': metrics_test
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Table Tennis Ball Detection Pipeline")
    parser.add_argument('--model', type=str, default="table-tennis-detection/weights.onnx", help="Path to weights (.onnx or .pt)")
    parser.add_argument('--mode', type=str, default="A", choices=['A', 'B', 'C'], help="Mode A (ROI pure), Mode B (ROI + MOG2 + KF), or Mode C (Pure KF + Static Filter)")
    parser.add_argument('--video', type=str, default="table-tennis-detection/videos/test_game_2.avi", help="Video path")
    parser.add_argument('--gt', type=str, default="table-tennis-detection/data_annotations/test_dataset_info.csv", help="GT CSV path")
    parser.add_argument('--save-video', type=str, default=None, help="Output visualization video path")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"Running Evaluate Pipeline: Model={args.model} | Mode={args.mode}")
    print(f"=======================================================")

    res = evaluate_video(
        model_path=args.model,
        video_path=args.video,
        gt_csv_path=args.gt,
        mode=args.mode,
        output_video=args.save_video
    )

    print(f"\nModel: {res['model']} | Mode: {res['mode']} | Pure Inference Speed: {res['fps_infer']} FPS")
    print(f"---------------------------------------------------------------------------------------------------")
    print(f"Scope                 | Recall@15px | Recall@25px | Precision@25px | F1@25px | MAE (px)")
    print(f"---------------------------------------------------------------------------------------------------")
    f_res = res['full']
    t_res = res['test']
    print(f"{f_res['scope']:21} | {f_res['recall_15px']:10.2f}% | {f_res['recall_25px']:10.2f}% | {f_res['precision_25px']:13.2f}% | {f_res['f1_25px']:6.2f}% | {f_res['mae_px']:6.2f} px")
    print(f"{t_res['scope']:21} | {t_res['recall_15px']:10.2f}% | {t_res['recall_25px']:10.2f}% | {t_res['precision_25px']:13.2f}% | {t_res['f1_25px']:6.2f}% | {t_res['mae_px']:6.2f} px")
    print(f"===================================================================================================\n")
