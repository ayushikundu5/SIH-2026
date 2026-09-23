"""Copy exactly the audio a second machine needs to train, and nothing else.

The corpora on disk are ~59 GB, but a training run only touches the files its
manifest names, and only in the train/val splits. This walks the manifest,
copies those files to a destination (a pendrive, an external disk) preserving
their layout under the data root, and copies the manifest beside them.

On the receiving machine the tree is restored anywhere and pointed at with
`SIH_DATA_ROOT`, which `audio.local_path` uses to rewrite the manifest's
Windows paths - so the manifest needs no editing and no rebuild.

    python scripts/export_training_data.py --dest E:/sih_data --dry-run
    python scripts/export_training_data.py --dest E:/sih_data

Test-split files are excluded on purpose: the receiving machine trains, it does
not evaluate (evaluation stays on the machine that owns the frozen test set, so
every number in the project comes from one instrument).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_ROOT = re.compile(r"^[A-Za-z]:[\\/]+SIH26052_data[\\/]+", re.I)


def collect(manifest: dict, splits: set[str]) -> set[str]:
    """Every `path` under the wanted splits. The split is whichever of
    train/val/test was the nearest enclosing key, so this works for speech,
    each noise category, the background pool and the RIRs alike."""
    out: set[str] = set()

    def walk(obj, split=None):
        if isinstance(obj, dict):
            if "path" in obj and isinstance(obj["path"], str):
                if split in splits:
                    out.add(obj["path"])
                return
            for k, v in obj.items():
                walk(v, k if k in ("train", "val", "test") else split)
        elif isinstance(obj, list):
            for v in obj:
                walk(v, split)

    walk(manifest)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifests/manifest_combat.json")
    ap.add_argument("--dest", required=True, help="e.g. E:/sih_data (pendrive)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--include", nargs="+", default=None,
                    help="only paths starting with these prefixes, relative to "
                         "the data root, e.g. --include prepared/ (the corpora "
                         "under raw/ are public downloads the other machine can "
                         "fetch itself; prepared/ holds our own conversions)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    man_path = ROOT / a.manifest
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    paths = sorted(collect(manifest, set(a.splits)))
    dest = Path(a.dest)

    total, missing, copied, skipped = 0, [], 0, 0
    for p in paths:
        rel = DATA_ROOT.sub("", p).replace("\\", "/")
        if a.include and not any(rel.startswith(pre) for pre in a.include):
            continue
        src = Path(p)
        if not src.exists():
            missing.append(p)
            continue
        size = src.stat().st_size
        total += size
        if a.dry_run:
            continue
        dst = dest / rel
        if dst.exists() and dst.stat().st_size == size:
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
        if copied % 2000 == 0:
            print(f"  copied {copied}/{len(paths)} files...", flush=True)

    print(f"files referenced by splits {a.splits}: {len(paths):,}")
    print(f"total size: {total / 2**30:.2f} GB")
    if missing:
        print(f"MISSING on this machine: {len(missing)} (first: {missing[0]})")
    if a.dry_run:
        print("dry run - nothing copied")
        return

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(man_path, dest / man_path.name)
    print(f"copied {copied} files, {skipped} already present")
    print(f"manifest -> {dest / man_path.name}")
    print(f"\nOn the other machine: restore this tree and run\n"
          f"  export SIH_DATA_ROOT=<where you put it>")


if __name__ == "__main__":
    main()
