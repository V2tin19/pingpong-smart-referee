# -*- coding: utf-8 -*-
"""离线增广 game1 训练集（路线A，零新依赖）。

只增广 train split；val 原样拷贝保持干净。三类增广：
  blur  ：沿球速方向的运动模糊（只模糊球patch， feathered streak mask 合成）
  light ：亮度/gamma 扰动，模拟曝光差
  paste ：把球（feathered streak mask 抠图）贴到 train 无球背景帧，原尺寸
输出 data/game1_dataset_aug + data.yaml + manifest.json。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import zipfile

import cv2
import numpy as np

ROI = dict(y_min=140, y_max=860, x_min=60, x_max=1860)
CROP_W = ROI["x_max"] - ROI["x_min"]   # 1800
CROP_H = ROI["y_max"] - ROI["y_min"]   # 720


def adaptive_wh(dx: float, dy: float):
    w = max(18.0, min(55.0, 16.0 + 0.85 * abs(dx)))
    h = max(18.0, min(55.0, 16.0 + 0.85 * abs(dy)))
    return w, h


def streak_mask(center, w, h, angle_deg, shape):
    """native 分辨率羽毛化拉丝掩码（椭圆沿速度方向）。"""
    m = np.zeros(shape[:2], np.float32)
    cv2.ellipse(m, (int(center[0]), int(center[1])),
                (max(2, int(w / 2)), max(2, int(h / 2))),
                angle_deg, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), 2.0)


def line_kernel(dx, dy, max_len=25):
    speed = (dx * dx + dy * dy) ** 0.5
    L = int(min(max_len, max(3, speed * 0.85))) | 1  # 奇数保证偏移对称不越界
    th = np.arctan2(dy, dx)
    k = np.zeros((L, L), np.float32)
    c = L // 2
    for i in range(L):
        x = int(c + (i - c) * np.cos(th))
        y = int(c + (i - c) * np.sin(th))
        k[y, x] = 1.0
    return k / k.sum()


def norm_label(cx, cy, w, h):
    return f"0 {cx / CROP_W:.6f} {cy / CROP_H:.6f} {w / CROP_W:.6f} {h / CROP_H:.6f}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/game1_dataset")
    ap.add_argument("--zip", default="E:/2026Autumn/602Lab/game_1.zip")
    ap.add_argument("--out", default="data/game1_dataset_aug")
    ap.add_argument("--n-blur", type=int, default=1600)
    ap.add_argument("--n-light", type=int, default=650)
    ap.add_argument("--n-paste", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    with zipfile.ZipFile(args.zip) as z:
        raw = json.loads(z.read("ball_markup.json").decode("utf-8"))
    markup = {int(k): (v["x"], v["y"]) for k, v in raw.items()
              if v and v.get("x", -1) >= 0 and v.get("y", -1) >= 0}

    def split_files(split):
        ld = os.path.join(args.src, "labels", split)
        pos, neg = [], []
        for f in sorted(os.listdir(ld)):
            fid = int(f[3:9])
            (pos if open(os.path.join(ld, f)).read().strip() else neg).append(fid)
        return pos, neg

    train_pos, train_neg = split_files("train")
    val_pos, val_neg = split_files("val")

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

    def copy_split(split, fids):
        for fid in fids:
            name = f"g1_{fid:06d}"
            shutil.copy2(os.path.join(args.src, "images", split, name + ".jpg"),
                         os.path.join(args.out, "images", split, name + ".jpg"))
            shutil.copy2(os.path.join(args.src, "labels", split, name + ".txt"),
                         os.path.join(args.out, "labels", split, name + ".txt"))

    copy_split("train", train_pos + train_neg)
    copy_split("val", val_pos + val_neg)

    manifest = []

    def save(split, name, img, label):
        cv2.imwrite(os.path.join(args.out, "images", split, name + ".jpg"),
                    img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        with open(os.path.join(args.out, "labels", split, name + ".txt"), "w") as f:
            f.write(label)

    def load(fid):
        return cv2.imread(os.path.join(args.src, "images", "train", f"g1_{fid:06d}.jpg"))

    def velocity(fid):
        p, n = markup.get(fid - 1), markup.get(fid + 1)
        c = markup[fid]
        if p and n:
            return (n[0] - p[0]) / 2, (n[1] - p[1]) / 2
        if n:
            return n[0] - c[0], n[1] - c[1]
        if p:
            return c[0] - p[0], c[1] - p[1]
        return 0.0, 0.0

    # blur + light：源为 train 正例
    for kind, n in (("blur", args.n_blur), ("light", args.n_light)):
        for i, fid in enumerate(random.sample(train_pos, min(n, len(train_pos)))):
            img = load(fid)
            gx, gy = markup[fid]
            cx, cy = gx - ROI["x_min"], gy - ROI["y_min"]
            dx, dy = velocity(fid)
            w, h = adaptive_wh(dx, dy)
            ang = np.degrees(np.arctan2(dy, dx))
            if kind == "blur":
                alpha = streak_mask((cx, cy), w, h, ang, img.shape)
                pad = int(max(w, h)) + 8
                x0, x1 = max(0, int(cx) - pad), min(CROP_W, int(cx) + pad)
                y0, y1 = max(0, int(cy) - pad), min(CROP_H, int(cy) + pad)
                patch = img[y0:y1, x0:x1]
                blurred = cv2.filter2D(patch, -1, line_kernel(dx, dy))
                a = alpha[y0:y1, x0:x1][..., None]
                img[y0:y1, x0:x1] = (blurred * a + patch * (1 - a)).astype(np.uint8)
            else:
                gamma = random.uniform(0.6, 1.5)
                lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)
                img = cv2.LUT(img, lut)
            name = f"g1aug_{kind}_{fid:06d}_{i:04d}"
            save("train", name, img, norm_label(cx, cy, w, h))
            manifest.append(dict(name=name, type=kind, src=fid, gamma=None if kind == "blur" else gamma))

    # paste：源正例球 -> train 无球背景
    for i in range(args.n_paste):
        src = random.choice(train_pos)
        tgt = random.choice(train_neg)
        simg, timg = load(src), load(tgt)
        gx, gy = markup[src]
        cx, cy = gx - ROI["x_min"], gy - ROI["y_min"]
        dx, dy = velocity(src)
        w, h = adaptive_wh(dx, dy)
        ang = np.degrees(np.arctan2(dy, dx))
        alpha = streak_mask((cx, cy), w, h, ang, simg.shape)
        pad = int(max(w, h)) + 8
        x0, x1 = max(0, int(cx) - pad), min(CROP_W, int(cx) + pad)
        y0, y1 = max(0, int(cy) - pad), min(CROP_H, int(cy) + pad)
        patch = simg[y0:y1, x0:x1].astype(np.float32)
        a = alpha[y0:y1, x0:x1][..., None]
        ph, pw = patch.shape[:2]
        px = random.randint(pw // 2 + 1, CROP_W - pw // 2 - 1)
        py = random.randint(ph // 2 + 1, int(CROP_H * 0.8) - ph // 2 - 1)
        tx0, ty0 = px - pw // 2, py - ph // 2
        region = timg[ty0:ty0 + ph, tx0:tx0 + pw].astype(np.float32)
        timg[ty0:ty0 + ph, tx0:tx0 + pw] = (patch * a + region * (1 - a)).astype(np.uint8)
        name = f"g1aug_paste_{src:06d}_{i:04d}"
        save("train", name, timg, norm_label(px, py, w, h))
        manifest.append(dict(name=name, type="paste", src=src, target=tgt, center=[px, py]))

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    yaml = (f"path: {os.path.abspath(args.out).replace(chr(92), '/')}\n"
            f"train: images/train\nval: images/val\n\nnames:\n  0: ball\n")
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(yaml)
    n_tr = len(os.listdir(os.path.join(args.out, "images", "train")))
    n_va = len(os.listdir(os.path.join(args.out, "images", "val")))
    print(f"train {n_tr} (orig {len(train_pos) + len(train_neg)} + aug {len(manifest)}), val {n_va} (clean)")
    print("manifest:", os.path.join(args.out, "manifest.json"))


if __name__ == "__main__":
    main()