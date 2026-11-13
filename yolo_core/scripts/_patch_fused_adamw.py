"""One-time workaround: torch 2.0.0 fused-AdamW DDP device bug.

Ultralytics 8.4.176 forces fused=True for AdamW on TORCH_2_0 (>=2.0.0), but
torch 2.0.0's fused AdamW leaves state_steps on CPU under DDP and crashes on the
first optimizer step (fixed in torch 2.0.1). Disable the fused kernel only on
2.0.0; non-fused AdamW is mathematically identical, just a touch slower. Keeps
the approved AdamW recipe so augmentation remains the only experimental variable.
Reversible: restores from trainer.py.bak_fused if present.
"""
from pathlib import Path

p = Path.home() / "pp_aug_ws/venv/lib/python3.10/site-packages/ultralytics/engine/trainer.py"
bak = p.with_suffix(".py.bak_fused")
old = ('fused = {"fused": True} if name in {"Adam", "AdamW"} and TORCH_2_0 '
       'and next(model.parameters()).is_cuda else {}')
new = ('fused = {"fused": True} if name in {"Adam", "AdamW"} and TORCH_2_0 '
       'and not torch.__version__.startswith("2.0.0") '
       'and next(model.parameters()).is_cuda else {}  # torch2.0.0 fused-AdamW DDP bug')

s = p.read_text(encoding="utf-8")
if new in s:
    print("ALREADY_PATCHED")
elif old in s:
    if not bak.exists():
        bak.write_text(s, encoding="utf-8")
    p.write_text(s.replace(old, new), encoding="utf-8")
    print("PATCHED (backup at %s)" % bak.name)
else:
    print("PATTERN_NOT_FOUND — inspect manually")
    raise SystemExit(1)