"""Step 5 of the real-combat data: add it to a COPY of the manifest.

Reads manifests/manifest.json and <prepared>/drive_combat/segments.json and
writes manifests/manifest_combat.json with one new noise category,
`combat_real`. The original manifest is never modified: the frozen test set and
every earlier training run reference it, and the frozen set is reproduced
byte-for-byte from it.

Segments Whisper flagged as containing speech (whisper_check_segments.py) are
dropped; the script refuses to run until every segment has been checked, since
an unchecked segment is exactly the kind that might teach the model that a
shouted voice is noise. Disjointness by video is asserted again here.

    python scripts/add_drive_to_manifest.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", default="C:/SIH26052_data/prepared/drive_combat/segments.json")
    ap.add_argument("--src", default="manifests/manifest.json")
    ap.add_argument("--out", default="manifests/manifest_combat.json")
    a = ap.parse_args()

    seg = json.loads(Path(a.segments).read_text(encoding="utf-8"))
    man = json.loads((ROOT / a.src).read_text(encoding="utf-8"))
    if "combat_real" in man.get("noise", {}):
        raise SystemExit(f"{a.src} already has combat_real - refusing to double-add")

    splits, report = {}, {}
    for k in ("train", "val", "test"):
        rows = seg["splits"][k]
        unchecked = [r for r in rows if "whisper" not in r]
        if unchecked:
            raise SystemExit(f"{len(unchecked)} {k} segments not Whisper-checked yet - "
                             "run scripts/whisper_check_segments.py first")
        kept = [r for r in rows if not r["whisper"]["speech"]]
        splits[k] = [{"path": str(Path(r["path"])), "group": r["group"], "source": r["source"],
                      "category": "combat_real", "type": r["type"], "dur": r["dur"]} for r in kept]
        report[k] = {"videos": len({r["group"] for r in kept}), "segments": len(kept),
                     "dropped_for_speech": len(rows) - len(kept),
                     "minutes": round(sum(r["dur"] for r in kept) / 60, 1)}

    g = {k: {r["group"] for r in v} for k, v in splits.items()}
    assert not (g["train"] & g["val"] or g["train"] & g["test"] or g["val"] & g["test"]), \
        "a video appears in more than one split"
    man["noise"]["combat_real"] = splits
    (ROOT / a.out).write_text(json.dumps(man), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print(f"wrote {a.out} (original {a.src} untouched)")


if __name__ == "__main__":
    main()
