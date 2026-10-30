"""
Build Mixed Table Tennis Dataset (Clean & Multi-Scene)
Fixes false-negative label noise:
1. Source 1: test_game_2.avi frames 0~389 (blue table, small target; both ROI crop and 1280x720 scaled)
2. Source 2: table-tennis-ball-detection dataset (green table, multi-angle, 75 images x 2 oversampled)
3. Source 3: test_rally.mp4 pure ceiling lights negative samples (y < 220 crop ONLY, no table/ball mislabeling)
"""

import os
import shutil
import cv2
import numpy as np
import pandas as pd

def build_clean_dataset(output_dir="mixed_tt_dataset",
                        game2_video="table-tennis-detection/videos/test_game_2.avi",
                        game2_csv="table-tennis-detection/data_annotations/test_dataset_info.csv",
                        rahimdzx_dir="table-tennis-ball-detection/Dataset",
                        rally_video="table-tennis-detection/test/test_rally.mp4"):

    output_dir = os.path.abspath(output_dir)
    # Remove existing to prevent any stale polluted files
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    train_img_dir = os.path.join(output_dir, "images", "train")
    val_img_dir = os.path.join(output_dir, "images", "val")
    train_lbl_dir = os.path.join(output_dir, "labels", "train")
    val_lbl_dir = os.path.join(output_dir, "labels", "val")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 65)
    print("Building Clean Multi-Scene Table Tennis Dataset (Zero-Leakage & Noise-Free)")
    print(f"Output Directory: {output_dir}")
    print("=" * 65)

    # ---------------------------------------------------------
    # 1. Source 1: test_game_2.avi frames 0~389 (Blue table)
    # ---------------------------------------------------------
    print("\n[Source 1] Extracting test_game_2.avi (Frames 0~389, Blue Table)...")
    df = pd.read_csv(game2_csv)
    df_game = df[df['game_id'] == 2].sort_values('frame_id').reset_index(drop=True)

    cap_g2 = cv2.VideoCapture(game2_video)
    roi_ymin, roi_ymax, roi_xmin, roi_xmax = 200, 850, 150, 1780
    crop_w = roi_xmax - roi_xmin  # 1630
    crop_h = roi_ymax - roi_ymin  # 650

    g2_roi_count = 0
    g2_scaled_count = 0

    for fid in range(390):
        ret, frame = cap_g2.read()
        if not ret:
            break

        row = df_game.iloc[fid]
        target_img_dir = val_img_dir if fid >= 312 else train_img_dir
        target_lbl_dir = val_lbl_dir if fid >= 312 else train_lbl_dir

        # 1A. Standard ROI crop
        crop_roi = frame[roi_ymin:roi_ymax, roi_xmin:roi_xmax]
        lines_roi = []
        if pd.notna(row['a']) and pd.notna(row['b']):
            orig_cx = float(row['a'] * 1920.0)
            orig_cy = float(row['b'] * 1080.0)
            crop_cx = orig_cx - roi_xmin
            crop_cy = orig_cy - roi_ymin
            if 0 <= crop_cx < crop_w and 0 <= crop_cy < crop_h:
                norm_cx = crop_cx / crop_w
                norm_cy = crop_cy / crop_h
                norm_bw = 20.0 / crop_w
                norm_bh = 20.0 / crop_h
                lines_roi.append(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_bw:.6f} {norm_bh:.6f}")

        cv2.imwrite(os.path.join(target_img_dir, f"g2_roi_f{fid:04d}.jpg"), crop_roi)
        with open(os.path.join(target_lbl_dir, f"g2_roi_f{fid:04d}.txt"), "w") as f:
            f.write("\n".join(lines_roi))
        g2_roi_count += 1

        # 1B. Full frame scaled to 1280x720 (learn multi-scale & full context)
        frame_720p = cv2.resize(frame, (1280, 720))
        lines_720p = []
        if pd.notna(row['a']) and pd.notna(row['b']):
            norm_cx = float(row['a'])
            norm_cy = float(row['b'])
            norm_bw = 16.0 / 1280.0
            norm_bh = 16.0 / 720.0
            lines_720p.append(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_bw:.6f} {norm_bh:.6f}")

        cv2.imwrite(os.path.join(target_img_dir, f"g2_720p_f{fid:04d}.jpg"), frame_720p)
        with open(os.path.join(target_lbl_dir, f"g2_720p_f{fid:04d}.txt"), "w") as f:
            f.write("\n".join(lines_720p))
        g2_scaled_count += 1

    cap_g2.release()
    print(f"  -> Generated {g2_roi_count} ROI crops + {g2_scaled_count} 720p scaled frames.")

    # ---------------------------------------------------------
    # 2. Source 2: Rahimdzx dataset (Green table, 2x oversampling)
    # ---------------------------------------------------------
    print("\n[Source 2] Incorporating Rahimdzx Dataset (Green Table, 2x Oversampled)...")
    r_train_img_dir = os.path.join(rahimdzx_dir, "train", "images")
    r_train_lbl_dir = os.path.join(rahimdzx_dir, "train", "labels")
    r_val_img_dir = os.path.join(rahimdzx_dir, "valid", "images")
    r_val_lbl_dir = os.path.join(rahimdzx_dir, "valid", "labels")

    r_count = 0
    # Process train with 2x copies
    for f in os.listdir(r_train_img_dir):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            stem = os.path.splitext(f)[0]
            lbl_file = os.path.join(r_train_lbl_dir, f"{stem}.txt")
            lbl_content = ""
            if os.path.exists(lbl_file):
                with open(lbl_file) as fp:
                    lbl_content = fp.read()

            for copy_idx in (1, 2):
                dst_img = f"rahim_cp{copy_idx}_{f}"
                dst_lbl = f"rahim_cp{copy_idx}_{stem}.txt"
                shutil.copyfile(os.path.join(r_train_img_dir, f), os.path.join(train_img_dir, dst_img))
                with open(os.path.join(train_lbl_dir, dst_lbl), "w") as fp:
                    fp.write(lbl_content)
                r_count += 1

    # Process valid (keep 1 copy for validation)
    for f in os.listdir(r_val_img_dir):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            stem = os.path.splitext(f)[0]
            lbl_file = os.path.join(r_val_lbl_dir, f"{stem}.txt")
            lbl_content = ""
            if os.path.exists(lbl_file):
                with open(lbl_file) as fp:
                    lbl_content = fp.read()

            shutil.copyfile(os.path.join(r_val_img_dir, f), os.path.join(val_img_dir, f"rahim_val_{f}"))
            with open(os.path.join(val_lbl_dir, f"rahim_val_{stem}.txt"), "w") as fp:
                fp.write(lbl_content)
            r_count += 1

    print(f"  -> Added {r_count} images & labels from Rahimdzx (2x oversampled in train).")

    # ---------------------------------------------------------
    # 3. Source 3: test_rally.mp4 (Pure ceiling light negative samples, y < 220 ONLY)
    # ---------------------------------------------------------
    print("\n[Source 3] Sampling test_rally.mp4 (Pure Ceiling Lights Negative Samples y < 220)...")
    cap_rally = cv2.VideoCapture(rally_video)
    rally_neg_count = 0
    frame_id = 0

    while cap_rally.isOpened() and rally_neg_count < 30:
        ret, frame = cap_rally.read()
        if not ret:
            break

        # Sample every 10 frames
        if frame_id % 10 == 0:
            # Crop ONLY ceiling lights (y < 220), strictly zero table, zero ball
            ceiling_crop = frame[0:220, :]
            target_img_dir = val_img_dir if rally_neg_count % 5 == 0 else train_img_dir
            target_lbl_dir = val_lbl_dir if rally_neg_count % 5 == 0 else train_lbl_dir

            img_name = f"rally_ceiling_f{frame_id:04d}.jpg"
            lbl_name = f"rally_ceiling_f{frame_id:04d}.txt"

            cv2.imwrite(os.path.join(target_img_dir, img_name), ceiling_crop)
            # Empty file: pure negative background
            open(os.path.join(target_lbl_dir, lbl_name), "w").close()
            rally_neg_count += 1

        frame_id += 1

    cap_rally.release()
    print(f"  -> Generated {rally_neg_count} pure ceiling light negative samples (strictly no table/ball mislabeling).")

    # ---------------------------------------------------------
    # 4. Generate data.yaml
    # ---------------------------------------------------------
    clean_output_dir = output_dir.replace('\\', '/')
    data_yaml_path = os.path.join(output_dir, "data.yaml")
    yaml_content = f"""# Ultralytics Clean Table Tennis Ball Dataset
path: {clean_output_dir}
train: images/train
val: images/val
nc: 1
names: ['ball']
"""
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    total_train = len(os.listdir(train_img_dir))
    total_val = len(os.listdir(val_img_dir))
    print("\n" + "=" * 65)
    print(f"Clean Dataset successfully built at: {output_dir}")
    print(f"Total Images: {total_train + total_val} (Train: {total_train}, Val: {total_val})")
    print(f"Config YAML: {data_yaml_path}")
    print("=" * 65)

if __name__ == '__main__':
    build_clean_dataset()
