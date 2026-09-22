"""Download the team's "Dataset" Drive folder: real combat footage (98 .mp4).

Source: the team's shared Drive folder (collected from YouTube, Aug-Sep 2026).
Its ID is deliberately NOT in the repo - pass it with --folder, or keep it in
C:/SIH26052_data/drive_folder_id.txt (the default). Graphic war footage from
third-party uploaders: used here as NOISE audio only, never shown.

Files are saved as `<drive_file_id>.mp4` - the titles are Arabic, Cyrillic and
emoji, which break tools on both Windows and Linux - with `index.json` mapping
id -> title. Resumable: a finished file gets a `.done` marker and is skipped.

Every download is checked for the MP4 signature (`ftyp` at byte 4): when Drive
rate-limits a file it answers 200 with an HTML page, and a size check alone
would happily store that page as a "video" (the same class of bug as the
datashare interstitial noted in CLAUDE.md).

    python scripts/download_drive_combat.py [--folder ID] [--out C:/SIH26052_data/raw/drive_combat]
"""
from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0"}


def list_folder(folder: str) -> list[dict]:
    url = f"https://drive.google.com/embeddedfolderview?id={folder}"
    s = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read().decode("utf-8", "replace")
    out = []
    for href, title in re.findall(
            r'<a href="([^"]+)"[^>]*>.*?<div class="flip-entry-title">(.*?)</div>', s, re.S):
        m = re.search(r"/file/d/([^/]+)", href)
        if m:
            out.append({"id": m.group(1), "title": html.unescape(title).strip()})
    return out


def fetch(item: dict, out: Path, tries: int = 6) -> str:
    dst = out / f"{item['id']}.mp4"
    done = dst.with_suffix(".mp4.done")
    if done.exists():
        return "skip"
    url = f"https://drive.usercontent.google.com/download?id={item['id']}&export=download&confirm=t"
    for attempt in range(tries):
        try:
            have = dst.stat().st_size if dst.exists() else 0
            hdr = dict(UA, **({"Range": f"bytes={have}-"} if have else {}))
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=120) as r:
                if "html" in (r.headers.get("Content-Type") or ""):
                    raise RuntimeError("got an HTML page (rate limit or permission), not the video")
                mode = "ab" if have and r.status == 206 else "wb"
                with open(dst, mode) as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            with open(dst, "rb") as f:
                head = f.read(12)
            if head[4:8] != b"ftyp":
                dst.unlink(missing_ok=True)
                raise RuntimeError("downloaded file is not an MP4")
            done.write_text(str(dst.stat().st_size))
            return f"ok {dst.stat().st_size / 2**20:.0f} MB"
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(min(300, 15 * 2 ** attempt))
    return f"FAILED: {err}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=None,
                    help="Drive folder ID (default: read C:/SIH26052_data/drive_folder_id.txt)")
    ap.add_argument("--out", default="C:/SIH26052_data/raw/drive_combat")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    folder = a.folder or Path("C:/SIH26052_data/drive_folder_id.txt").read_text().strip()
    items = list_folder(folder)
    (out / "index.json").write_text(json.dumps(
        {"source": f"https://drive.google.com/drive/folders/{folder}", "items": items},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(items)} videos listed -> {out}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        for n, (it, res) in enumerate(zip(items, ex.map(lambda i: fetch(i, out), items)), 1):
            print(f"[{n:3d}/{len(items)}] {res:22s} {it['id']}", flush=True)
    failed = [i for i in items if not (out / f"{i['id']}.mp4.done").exists()]
    print(f"done: {len(items) - len(failed)} ok, {len(failed)} failed - re-run to retry", flush=True)


if __name__ == "__main__":
    main()
