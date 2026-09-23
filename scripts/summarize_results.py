"""One compact table of every model measured, across all three test sets.

Reads the per-clip CSVs `src/evaluate.py` writes and prints PESQ-WB / STOI /
output SNR per model, plus the problem-statement verdict for a chosen model.
Nothing is computed here that evaluate.py did not measure - this only groups
what is already on disk, so it can be re-run any time and cannot invent a
number.

    python scripts/summarize_results.py
    python scripts/summarize_results.py --highlight onnx:artifacts/model_fresh32_simple.onnx
    python scripts/summarize_results.py --oneline    # for a phone notification
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SETS = [("onnx", "frozen defence (720)"),
        ("combat", "real combat (150)"),
        ("vbd_onnx", "VoiceBank-DEMAND (824)")]

NICE = {
    "unprocessed": "no cleaning",
    "wiener": "Wiener filter",
    "onnx:artifacts/gtcrn_dns3_simple.onnx": "GTCRN pretrained",
    "onnx:artifacts/model_simple.onnx": "shipped w16",
    "onnx:artifacts/model_wide32_simple.onnx": "wide32",
    "onnx:artifacts/model_combat32_simple.onnx": "combat32",
    "onnx:artifacts/model_fresh32_simple.onnx": "fresh32",
    "onnx:artifacts/model_wide48_simple.onnx": "wide48",
}
TARGETS = {"pesq": 2.5, "stoi": 0.85, "snr": 15.0}


def load(tag: str) -> pd.DataFrame | None:
    p = ROOT / "results" / f"per_clip_{tag}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d["model"] = d.method.map(lambda m: NICE.get(m, m))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--highlight", default="onnx:artifacts/model_fresh32_simple.onnx")
    ap.add_argument("--oneline", action="store_true",
                    help="a single line for a phone notification")
    a = ap.parse_args()

    frames = {name: load(tag) for tag, name in SETS}
    hi = NICE.get(a.highlight, a.highlight)

    if a.oneline:
        bits = []
        for name, d in frames.items():
            if d is None or hi not in set(d.model):
                continue
            r = d[d.model == hi][["pesq", "stoi", "snr"]].mean()
            bits.append(f"{name.split(' (')[0]}: PESQ {r.pesq:.3f} STOI {r.stoi:.3f} SNR {r.snr:.1f}dB")
        print(f"{hi} | " + " | ".join(bits) if bits else f"{hi}: not evaluated yet")
        return

    for name, d in frames.items():
        if d is None:
            print(f"\n=== {name}: not evaluated")
            continue
        g = (d.groupby("model")[["pesq", "pesq_nb", "stoi", "snr", "snr_gain"]]
               .mean().round(3).sort_values("pesq"))
        print(f"\n=== {name}")
        print(g.to_string())

    d = frames.get("frozen defence (720)")
    if d is not None and hi in set(d.model):
        r = d[d.model == hi][["pesq", "stoi", "snr"]].mean()
        print(f"\n=== problem-statement targets, {hi}, frozen defence set")
        for k, label in (("pesq", "PESQ > 2.5"), ("stoi", "STOI > 0.85"), ("snr", "output SNR > 15 dB")):
            v = float(r[k])
            ok = "PASS" if v > TARGETS[k] else f"short by {TARGETS[k] - v:.3f}"
            print(f"  {label:22s} {v:7.3f}   {ok}")


if __name__ == "__main__":
    main()
