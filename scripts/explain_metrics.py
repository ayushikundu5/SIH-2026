"""Show, clip by clip, exactly how PESQ / STOI / SNR are measured.

Built for screen-sharing: every step prints what it does, and the scores come
from the same reference implementations `src/evaluate.py` uses for the full
720-clip table (`pesq` = ITU-T P.862, `pystoi`), called through the same
`src/metrics.py` functions - so the numbers here are produced exactly the way
the official results were.

For each clip it:
  1. loads the CLEAN speech and the NOISY mixture from the frozen test set
  2. runs the NOISY audio through the shipped ONNX model in 16 ms chunks,
     exactly as the hardware will (no PyTorch)
  3. scores both the noisy input and the model's output against the clean
     speech, so the "before" and "after" sit side by side
  4. writes clean / noisy / enhanced wavs so the audio can be played

    python scripts/explain_metrics.py                    # 3 clips, one per category
    python scripts/explain_metrics.py --category gunshot --n 5
    python scripts/explain_metrics.py --id gunshot/0000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import audio as A                  # noqa: E402
from src import metrics as M                # noqa: E402
from src.methods import enhance_onnx        # noqa: E402

TARGETS = {"pesq": 2.5, "stoi": 0.85, "snr": 15.0}


def mark(name: str, value: float) -> str:
    return "PASS" if value > TARGETS[name] else "below target"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/model_simple.onnx")
    ap.add_argument("--testset", default=r"C:\SIH26052_data\testset")
    ap.add_argument("--category", default=None,
                    help="gunshot | artillery | rotor | engine | siren | babble")
    ap.add_argument("--id", default=None, help="one clip, e.g. gunshot/0000")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/explain_metrics")
    a = ap.parse_args()

    ts = Path(A.local_path(a.testset))
    index = json.loads((ts / "index.json").read_text(encoding="utf-8"))
    items = index["items"]
    if a.id:
        items = [it for it in items if it["id"] == a.id]
    elif a.category:
        items = [it for it in items if it["category"] == a.category]
        items = [items[i] for i in np.random.default_rng(a.seed).choice(
            len(items), min(a.n, len(items)), replace=False)]
    else:
        cats = sorted({it["category"] for it in items})
        rng = np.random.default_rng(a.seed)
        pick = rng.choice(cats, min(a.n, len(cats)), replace=False)
        items = [rng.choice([it for it in items if it["category"] == c]) for c in pick]
    if not items:
        raise SystemExit("no matching clips")

    model = ROOT / a.model
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"test set : {ts}  ({len(index['items'])} frozen clips, seed {index['seed']})")
    print(f"model    : {a.model}  (ONNX, streamed in 16 ms chunks)")
    print("metrics  : pesq.pesq(mode='wb') = ITU-T P.862.2 | pystoi.stoi | "
          "SNR = 10*log10(sum(clean^2) / sum((output-clean)^2))")

    rows = []
    for it in items:
        clean = A.load_audio(ts / it["clean"], 16000)
        noisy = A.load_audio(ts / it["noisy"], 16000)
        t0 = time.perf_counter()
        enh = enhance_onnx(noisy, 16000, model)
        dt = time.perf_counter() - t0

        before = {"pesq": M.pesq_wb(clean, noisy), "stoi": M.stoi_score(clean, noisy),
                  "snr": M.snr_db(clean, noisy)}
        after = {"pesq": M.pesq_wb(clean, enh), "stoi": M.stoi_score(clean, enh),
                 "snr": M.snr_db(clean, enh)}
        rows.append((before, after))

        stem = it["id"].replace("/", "_")
        A.save_audio(out_dir / f"{stem}_1_clean.wav", clean)
        A.save_audio(out_dir / f"{stem}_2_noisy.wav", noisy)
        A.save_audio(out_dir / f"{stem}_3_model_output.wav", enh)

        dur = len(noisy) / 16000
        print(f"\n--- clip {it['id']}  ({it['category']}, {dur:.1f} s, "
              f"{it.get('n_events', 0)} burst(s), processed in {dt:.2f} s = "
              f"{dt/dur:.2f}x real time)")
        print(f"  {'':6s} {'noisy input':>12s} {'model output':>13s}   target")
        print(f"  {'PESQ':6s} {before['pesq']:12.2f} {after['pesq']:13.2f}   > 2.5   "
              f"{mark('pesq', after['pesq'])}")
        print(f"  {'STOI':6s} {before['stoi']:12.3f} {after['stoi']:13.3f}   > 0.85  "
              f"{mark('stoi', after['stoi'])}")
        print(f"  {'SNR':6s} {before['snr']:9.1f} dB {after['snr']:10.1f} dB   > 15 dB "
              f"{mark('snr', after['snr'])}")
        print(f"  listen: {out_dir.relative_to(ROOT)}\\{stem}_[1_clean|2_noisy|3_model_output].wav")

    if len(rows) > 1:
        b = {k: np.mean([r[0][k] for r in rows]) for k in TARGETS}
        f = {k: np.mean([r[1][k] for r in rows]) for k in TARGETS}
        print(f"\n=== average of these {len(rows)} clips ===")
        print(f"  PESQ {b['pesq']:.2f} -> {f['pesq']:.2f}   STOI {b['stoi']:.3f} -> "
              f"{f['stoi']:.3f}   SNR {b['snr']:.1f} -> {f['snr']:.1f} dB")
    print("\nThe official numbers are this same calculation over all 720 clips: "
          "results/per_clip_onnx.csv (one row per clip), summarised in "
          "results/results_onnx.md")


if __name__ == "__main__":
    main()
