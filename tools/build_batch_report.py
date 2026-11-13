"""Assemble the 7-video batch evaluation into a facts-only markdown report.

Reads benchmark reports (baseline vs candidate) and referee logs produced by
tools/run_batch_referee.sh. Emits observed numbers and configuration/caveats;
no promotional conclusions.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "output" / "batch_20261113"
PRESET = {1: "game2*", 2: "test2", 3: "test3", 4: "test4", 5: "test3", 6: "test3", 7: "game2*"}


def metrics(path):
    if not path.exists():
        return None
    rep = json.loads(path.read_text(encoding="utf-8"))
    r = next(x for x in rep["results"] if x["imgsz"] == 960)
    return r["metrics"]


def f(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}"


def log_facts(i):
    p = BATCH / "logs" / f"test{i}_referee.log"
    if not p.exists():
        return {}
    txt = p.read_text(encoding="utf-8", errors="replace")
    out = {}
    m = re.search(r"【传统状态机比分】: P1 \(半台A\) (\d+) : (\d+) P2", txt)
    if m:
        out["legacy"] = f"{m.group(1)}:{m.group(2)}"
    m = re.search(r"【回合证据引擎比分】: P1 \(半台A\) (\d+) : (\d+) P2 \(半台B\) \(已裁决 (\d+) 回合\)", txt)
    if m:
        out["evidence"] = f"{m.group(1)}:{m.group(2)}"
        out["rallies"] = m.group(3)
    m = re.search(r"处理帧数: (\d+) \| 耗时: ([\d.]+)s \(([\d.]+) FPS\)", txt)
    if m:
        out["frames"], out["sec"], out["fps"] = m.group(1), m.group(2), m.group(3)
    m = re.search(r"物理事件统计: (\{.*\})", txt)
    if m:
        out["events"] = m.group(1)
    return out


def main():
    lines = []
    lines.append("# 七视频批量评测结果（2026-11-13）\n")
    lines.append("模型：基线 `weights/yolo11n_game1_best.pt` vs 候选/正式 `weights/yolo11n_game1_aug_best.pt`（game1 离线增广重训）。\n")
    lines.append("检测指标：`tools/benchmark_input_sizes.py`，imgsz=960，conf=0.18，生产适配器+ROI，稀疏点标注。")
    lines.append("状态机视频：`run_referee.py`，imgsz=768、720p(0.667)、60fps（降质加速），adjudicator=shadow（HUD 显示传统状态机，日志含双引擎）。\n")
    lines.append("机位预设：test_2/3/5/6=game2/test3 同机位；test_4=test4；**test_1/test_7 无精确标定，回退 game2（*近似）**，其状态机比分/落点仅供参考，检测召回不受影响（ROI 仍覆盖球）。\n")
    lines.append("旧结果已归档至 `output/_archive_20261113/`；本批产物在 `output/batch_20261113/`。\n")

    lines.append("## 一、检测指标对照（960）\n")
    lines.append("| 视频 | 权重 | 正样本帧 | Recall@5px | Recall@25px | MAE(px) | RMSE(px) | 不可见帧检出 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for i in range(1, 8):
        for tag, label in (("baseline", "基线"), ("candidate", "增广")):
            m = metrics(BATCH / "metrics" / f"test{i}_{tag}.json")
            if m is None:
                lines.append(f"| test_{i} | {label} | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| test_{i} | {label} | {m['positive_frames']} | {f(m['recall_5px_pct'])} | "
                f"{f(m['recall_25px_pct'])} | {f(m['mean_center_error_px'])} | {f(m['rmse_px'])} | "
                f"{m['detections_on_annotated_invisible']}/{m['annotated_invisible_frames']} |")
    lines.append("")

    lines.append("## 二、状态机运行与比分（视频见 videos/）\n")
    lines.append("| 视频 | 预设 | 处理帧 | 耗时(s) | 处理FPS | 传统状态机比分 | 证据引擎比分(裁决回合) |")
    lines.append("|---|---|---|---|---|---|---|")
    for i in range(1, 8):
        d = log_facts(i)
        lines.append(
            f"| test_{i} | {PRESET[i]} | {d.get('frames','—')} | {d.get('sec','—')} | {d.get('fps','—')} | "
            f"{d.get('legacy','—')} | {d.get('evidence','—')}({d.get('rallies','—')}) |")
    lines.append("")
    lines.append("注：比分为状态机在该视频上的观测输出，非人工标注真值；test_1/test_7 因机位近似，比分不作准。")
    lines.append("事件统计逐视频见 `logs/testN_referee.log` 末尾「物理事件统计」。\n")

    out = ROOT / "docs" / "batch_eval_20261113.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("report:", out)


if __name__ == "__main__":
    main()