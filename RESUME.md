# Resume here

Current state, what is blocking, and exactly how to pick each item up.
`CLAUDE.md` has the architecture and the invariants; `README.md` has results,
setup, and where to hear every audio file.

---

## In one screen — state at 23 Sep 2026, 16:30 IST

**Running now: `fresh32`.** Width 32 from scratch, 170 epochs, and the first run
whose training mixtures actually change every epoch (invariant 14a was only
fixed on 22 Sep). Started 13:07, epoch 33 at 16:25, ~5.4 min/epoch, ETA ~04:40
on 24 Sep. Log: `C:\SIH26052_data\train_fresh32.log`.

Nobody needs to be present for the rest of it. The job was launched through WMI,
so it does not belong to any editor or Claude session and survives them closing;
a watchdog kills and resumes it if CUDA hangs; and `auto_finish_fresh32.ps1`
then exports the best checkpoint to streaming ONNX, scores it on all three test
sets, benchmarks real-time speed and pushes the headline to the phone by itself.

**Armed behind it: `short24`** (`chain_short24.ps1`, running since 16:34). It
waits for the finished marker, for the GPU to be free and for the evaluation to
stop, then starts the 320/160 run described below with the same three helpers,
and runs `finish_short24.ps1` when that ends. The GPU would otherwise sit idle
from about 04:40. **To cancel it, before or during:**

```powershell
New-Item C:\SIH26052_data\STOP_SHORT24 -ItemType File
```

It writes only its own checkpoints, log and artifacts - nothing `fresh32`
produced, nothing shipped, and not the frozen test set.

**Where the GitHub clone lives:** `C:\SIH26052_data\repo_push` (moved out of a
session temp directory on 23 Sep, where it would eventually have been deleted).
The project directory itself is deliberately NOT a git repo - 64 GB of data and
a 47k-file venv sit beside it. Publishing is: copy the changed files into that
clone, commit, `git push origin main`. It is cloned with `core.autocrlf=false`,
which matters: the user's global setting is `true`, and a clone made with it
rewrites every `.sh` file with CRs that WSL cannot run. Watch for the same trap
when writing files from Python on Windows - `write_text` turns `\n` into CRLF
and produces a diff that touches every line of the file.

**Best FINISHED model:** `combat32` — `artifacts/model_combat32_simple.onnx`,
`checkpoints/combat32_best.pt`. Width 32 (103,381 params), trained on defence
mixtures + real combat audio. Best of every model on all three test sets.

| | PESQ-WB | STOI | out SNR | vs target |
|---|---|---|---|---|
| frozen defence set (720) | 2.106 | 0.876 | 11.72 dB | STOI ✅, PESQ −0.39, SNR −3.3 dB |
| real combat set (150) | 1.977 | 0.865 | 12.15 dB | STOI ✅ |
| VoiceBank-DEMAND (824) | 2.334 | 0.911 | 17.52 dB | STOI ✅, **SNR ✅** |

All three targets pass at input SNR ≥ 10 dB (2.83 / 0.96 / 17.4 dB).
RTF 0.476 ✅ (< 0.5); latency 40.99 ms ✗ at 512/256 — see immediately below.

### The 32 ms latency target is reachable after all (new, 23 Sep afternoon)

The old conclusion — "the latency target cannot be met" — was correct about the
512-point window and wrong as a general statement, and the distinction matters
because latency is one of the deliverables. Algorithmic delay is chunk buffering
(hop) plus the overlap-add delay (win − hop), which sums to **the window
length**, 32.00 ms, before any compute. So no model size can meet the target at
512/256. A shorter window can.

`src/framing.py` now takes the size from the environment (`SIH_NFFT`, `SIH_HOP`;
**unset means the shipped 512/256, so nothing that exists changes**), and the
band arithmetic, the ERB split, the dual-path RNN width, the streaming cache
shapes, the ONNX export and the live runtime all follow it. Measured on this
laptop, ONNX streaming, 1 thread, **while the machine was busy training** — so
these are an upper bound, idle is about 0.74× (`results/bench_framing_probe.json`):

| framing | width | params | p95 compute | total latency | RTF |
|---|---|---|---|---|---|
| 512/256 | 32 | 103,381 | 12.73 ms | 44.73 ms ✗ | 0.646 |
| 320/160 | 32 | 85,333 | 8.97 ms | **28.90 ms ✅** | 0.716 |
| **320/160** | **24** | **54,597** | **7.41 ms** | **27.41 ms ✅** | **0.533** |
| 320/160 | 16 | 31,733 | 6.41 ms | 26.41 ms ✅ | 0.489 ✅ |

The 320-point rows are **untrained probe exports** — per-frame time depends on
the graph, not on the weights in it, so speed transfers and quality does not
exist in those rows. The overlap-add delay was measured by cross-correlation at
159-160 samples = win − hop, confirming the streaming path is genuinely correct
at the new size, not merely running.

Width 24 is the pick: latency passes with 4.6 ms of margin and RTF scales to
~0.39 idle, inside the 0.5 budget with room for a slower embedded CPU. Width 32
at this window does not fit (≈0.53 idle), the same trade that ruled out width 48
at the long window.

The whole path was exercised end to end at 320/160 before being armed, on a
one-epoch throwaway model: training runs and validates (54,597 params, "10 ms
chunks" printed at startup), the ONNX export **MATCHES the offline model**
frame by frame, the evaluator scores it through `StreamingEnhancer`, and the
same model **refuses to run** with the framing unset -

> `smoke_short24_simple.onnx expects 161 frequency bins (n_fft=320) but this
> process is framing at n_fft=512. Set SIH_NFFT=320 (and SIH_HOP).`

- which is the one failure mode that would otherwise have produced confident
nonsense. The throwaway checkpoint, its export and its CSVs were deleted.

**What is NOT known yet:** what the shorter window costs in quality. 161 bins
instead of 257 and a 21-wide band axis instead of 33 is coarser frequency
resolution, and PESQ/STOI/word score at 320/160 have never been measured.
`configs/train_short24.yaml` is ready; the chain above launches it, or by hand:

```powershell
$env:SIH_NFFT=320; $env:SIH_HOP=160
& $PY -m src.train --config configs/train_short24.yaml --tag short24
```

Both variables are needed by every downstream step too (export, evaluate,
bench). Forgetting them cannot corrupt a result quietly: `export_onnx` reads the
window size back out of the checkpoint weights and refuses a mismatch, and
`StreamingEnhancer` refuses a graph whose bin count disagrees with its own STFT.

**Open decisions, in priority order:**
1. **Swap the deliverable to `combat32`?** `artifacts/model.onnx`, `SPEC.md`
   and `example_inference.py` still describe the w16 model. One command:
   `python scripts/make_handoff.py --model artifacts/model_combat32_simple.onnx`.
   Worth waiting for `fresh32` to land first.
2. **Re-run the ASR intelligibility test** (`scripts/asr_score.py --model
   medium`) on `combat32`. The "suppression kills words" finding is measured on
   the w16 model only; it is the project's most important open question.
3. **Launch `short24`** once `fresh32` finishes, and report latency and quality
   TOGETHER. A model that meets the 32 ms target and loses words is not an
   improvement.
4. **Retrain `wide32` with the 14a fix alone**, to separate "real data" from
   "training data that actually changes" in the `combat32` result. `fresh32`
   answers most of this.
5. Push the latest work to the teammate's repo branch
   (`Babanstar456/SIH-2026-main`, branch `wide-gtcrn-onnx-eval`) — it still
   stops at 21 Sep. The user's own repo `ayushikundu5/SIH-2026` is current.
6. Power setting is back to the 15-minute default on AC (reverted 23 Sep); a
   keep-awake process holds the machine up for as long as training runs.

**Deadline context:** SIH PPT submission 23 Sep. The GitHub link that goes with
it is `ayushikundu5/SIH-2026` — **private**, so it must be made public or the
evaluators added before submitting.

---

## 21 Sep 2026 — merge + where the three PS targets stand

Teammate repo (commit `23bf15b`) merged into the Windows copy. Goal now: the
problem statement's targets — PESQ > 2.5, STOI > 0.85, SNR > 15 dB — before
the SIH PPT round.

**Measured, torch-free, via the shipped ONNX exports** (`onnx:<path>` method,
validated against the torch path: PESQ 1.931 vs 1.933 on the same 720 clips).
`results/results_onnx.md`, `results/results_vbd_onnx.md`:

| model | frozen defence set PESQ-WB / STOI / output SNR | VoiceBank-DEMAND PESQ-WB / STOI / output SNR |
|---|---|---|
| unprocessed | 1.319 / 0.801 / 4.41 dB | 1.968 / 0.921 / 8.45 dB |
| gtcrn_dns3 (pretrained) | 1.814 / 0.854 / 10.07 dB | 2.511 / 0.915 / 15.87 dB |
| **shipped** | **1.931 / 0.860 / 10.79 dB** | 2.387 / 0.921 / 16.34 dB |
| lowsnr | 1.854 / 0.851 / 10.18 dB | 2.420 / 0.919 / 16.82 dB |

- **STOI passes; PESQ (−0.57) and SNR (−4.2 dB output) do not** on the defence set.
- **`lowsnr` is worse than shipped on the frozen set on every metric**
  (paired, 720 clips: PESQ −0.078, STOI −0.009, SNR −0.60 dB, all p < 1e-50),
  including the < 0 dB input band it was trained for. Its claimed advantage is
  on real recordings via ASR — a different instrument; not contradicted, but it
  does not help the PS targets.
- **Capacity is the ceiling.** Three differently-trained 48K-param GTCRNs land
  at PESQ 1.81–1.93; val PESQ in `train_log_ft.csv` moves 1.779 → 1.788 over 60
  epochs while train loss keeps falling. More training of the same model will
  not close 0.57 PESQ. Next lever: a wider GTCRN (width is hardcoded to 16 in
  `src/models/gtcrn.py`, the stream export and `framing.py` cache shapes).
- **The SNR target has two readings.** The code scored SNR *gain* > 15 dB, which
  is unreachable here (median input +4 dB). "Output SNR > 15 dB" is the reading
  parallel to STOI/PESQ. Tables now show both, labelled. Choice is the team's.
- **PESQ-NB added alongside WB** (shipped: 2.495 on the defence set). Pass/fail
  stays on wideband.
- **Live demo of how the numbers are made:** `scripts/explain_metrics.py`
  (`--category gunshot --n 5`, `--id gunshot/0000`) scores a few frozen-set
  clips before/after through the shipped ONNX with the same `metrics.py`
  calls as `evaluate.py`, and writes clean/noisy/output wavs to
  `results/explain_metrics/`. Cross-checked: identical to `per_clip_onnx.csv`.
- **Suspect row in `results/method_comparison.md`:** `_rnnoise` never
  compensates RNNoise's 10 ms frame delay; −32.77 dB SI-SDR matches the
  −32.55 dB an unaligned comparison produces (invariant 10). Re-measure before
  claiming "RNNoise/DeepFilterNet lose to unprocessed".

**Torch on the Windows machine: native again as of 23 Sep** — `import torch`
succeeds with `cuda.is_available() == True` and all 36 tests pass natively,
although Smart App Control still reports enabled, so this is reputation-based
and may flip back. Native is also FASTER than WSL for training (0.26 s/step
measured natively vs 0.39 s/step in WSL, which reads the dataset over the 9P
bridge unless it is mirrored to `/root/sih/data`). **Check first**, then choose:

```powershell
& "C:\SIH26052_data\.venv\Scripts\python.exe" -c "import torch; print(torch.cuda.is_available())"
```

The WSL2 route below stays valid as the fallback (and is what `combat32` was
trained with). Smart App Control blocked `torch.dll` natively on 21-22 Sep
(WinError 4551), so training/export ran in WSL2
Ubuntu-24.04 with its own venv at `/root/sih/.venv` (torch 2.13.0+cu126, sees
the RTX 3050). Manifest paths (`C:\...`) are mapped to `/mnt/c/...` by
`audio.local_path`. From Git Bash (`MSYS_NO_PATHCONV=1` stops Git Bash
rewriting the Linux paths; `wsl.exe` also swallows `$vars` in `bash -c`, so use
script files):

```bash
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 -u root -- bash /mnt/c/SIH26052_data/wsl_run.sh -m pytest tests -q
```

Windows-native still does ONNX eval: `evaluate.py --workers 11` with `onnx:`
methods, ~3 min per model for the 720-clip set.

### !! Training-data bug found 22 Sep 16:20 — read before trusting any plateau

Every training run until now saw ONE fixed set of `epoch_size` mixtures,
repeated every epoch (persistent DataLoader workers never received
`set_epoch`; see CLAUDE.md invariant 14a). Fixed in `src/train.py`, guarded by
`tests/test_train_loader.py`. Consequences:
- Evaluation numbers of existing models are still valid measurements of those
  models - they are simply under-trained, not mis-measured.
- The "capacity ceiling" reading of the flat val curves (below, and in the
  wide32 section) is CONFOUNDED: a model memorising 20k fixed mixtures also
  goes flat. Re-test before repeating it, e.g. a fixed-data wide32 run.

### Done: `combat32` — wide32 fine-tuned with REAL combat noise

Started 22 Sep 2026 14:20 IST, 60 epochs (~6 h + lid-closed time). Epochs 0-19
ran with the frozen-data bug (epoch-0 set for 0-14, epoch-15 set for 15-19 after
a GPU hang at 15:44 forced a restart); resumed at epoch 20 at 16:26 WITH the
fix, so epochs 20-59 each see fresh mixtures.

**(Re)start everything with `C:\SIH26052_data\start_combat32.ps1`**
(right-click > Run with PowerShell; safe to re-run - it skips whatever is
already running). It starts `run_combat32_keep_awake.ps1` ->
`train_combat32_loop.sh` (self-healing, `--resume`), `train_watchdog_v2.sh`
(kills a hung run: at once on a logged "CUDA error", else after >5 min of
silence - WSL2 threw "CUDA error: unknown error" and HUNG twice on 22 Sep, at
15:46 and 22:37) and the ntfy notifier through WMI (`Win32_Process.Create`), so they
do NOT die with Claude/VS Code. Jobs started from a Claude session - even with
`Start-Process` - sit in that session's Windows job object and are killed when
it closes; that stopped training twice on 22 Sep (16:29, 17:49; resumed at
epochs 20 and 32). Stop file `C:\SIH26052_data\STOP_COMBAT32`; log
`C:\SIH26052_data\train_combat32.log`.

The data - the team's Drive folder of 98 combat videos (YouTube, Syria /
Ukraine), turned into speech-free noise by:

1. `scripts/download_drive_combat.py` - 98 mp4, 4.2 GB (folder ID kept out of
   the repo: `C:\SIH26052_data\drive_folder_id.txt`)
2. `scripts/extract_drive_audio.py` - 96 with audio, 12.7 h, 16 kHz mono
3. `scripts/tag_drive_audio.py` (WSL, PANNs Cnn14) - every 1 s window scored
   on 527 AudioSet classes. **Median speech score 0.47: over half has voices.**
4. `scripts/select_drive_segments.py` - keep speech<0.1 & music<0.1 & >-50 dBFS,
   trim 0.5 s each edge, >=3 s; split BY VIDEO (compilations -> train)
5. `scripts/whisper_check_segments.py` (WSL GPU) - independent speech check:
   6/1454 flagged, all hallucinations ("Thank you.", "BOOM!"), dropped anyway
6. `scripts/add_drive_to_manifest.py` -> `manifests/manifest_combat.json`
   (original manifest untouched); `configs/data_combat.yaml` adds
   `combat_real` (steady layer, weight 2.0); `configs/train_combat.yaml`

Result: train 68 videos / 138 min, val 4 / 10 min, test 21 / 26 min
(gunfire 55%, vehicle 20%, explosion 8%, ambient 27%; aircraft did not survive
the speech filter). `make_testset.py` gained `--config/--manifest/--categories/
--no-background` (defaults verified byte-identical on the frozen set) and
rendered **`C:\SIH26052_data\testset_combat`**: 150 clips, held-out speakers +
real combat noise only, realised input SNR median +4.8 dB (-1.4..+15.6).

**FINISHED 23 Sep 00:27, 60/60 epochs** (best = epoch 56, val PESQ 2.1014;
val_pesq is NOT comparable with earlier runs - the new category changed the val
mixtures). Survived three WSL "CUDA error: unknown error" hangs (15:46, 22:37,
00:00) and two Claude-session kills; ~20 min lost in total. Exported
`artifacts/model_combat32{,_simple}.onnx` (streaming vs offline max abs diff
8.8e-7, `results/onnx_verify_combat32.json`). **RTF 0.4761** (target < 0.5,
idle machine, `results/bench_edge_combat32.json`); latency 40.99 ms vs the
32 ms target, as expected for width 32.

**combat32 is the best model on ALL THREE test sets** (paired vs wide32, all
p < 1e-10):

| | frozen defence (720) | real combat (150) | VoiceBank-DEMAND (824) |
|---|---|---|---|
| | PESQ / STOI / out SNR | PESQ / STOI / out SNR | PESQ / STOI / out SNR |
| unprocessed | 1.319 / 0.801 / 4.41 | 1.298 / 0.783 / 5.72 | 1.968 / 0.921 / 8.45 |
| shipped (w16) | 1.931 / 0.860 / 10.79 | 1.858 / 0.850 / 11.46 | 2.387 / 0.921 / 16.34 |
| wide32 | 2.057 / 0.873 / 11.44 | 1.879 / 0.860 / 11.94 | 2.111 / 0.903 / 15.91 |
| **combat32** | **2.106 / 0.876 / 11.72** | **1.977 / 0.865 / 12.15** | **2.334 / 0.911 / 17.52** |

- vs wide32: defence PESQ +0.050 (78% of clips), real-combat PESQ +0.098 (81%),
  VBD PESQ +0.223 (73%) and VBD SNR +1.62 dB (82%).
- **The VBD regression is largely repaired** (2.111 -> 2.334 vs shipped's
  2.387) and VBD output SNR 17.52 dB now beats shipped's 16.34.
- Frozen set by input SNR: <0 dB 1.48/0.76/6.3, 0-5 1.89/0.87/10.3,
  5-10 2.29/0.92/13.3, **10-15 2.83/0.96/17.4**, **>15 3.18/0.97/20.2** - all
  three PS targets pass at >= 10 dB input.
- Burst SI-SDR gain: gunshot 8.39 dB, artillery 8.78 (shipped: 7.16 / 7.90).
- **Attribution caveat:** this run changed TWO things at once - the real-combat
  category AND the frozen-data fix (epochs 20-59). A wide32 retrain with only
  the fix would separate them; not run.
- Still short of PESQ 2.5 (-0.39) and output SNR 15 dB (-3.3) as a whole-set
  average on the defence set. `artifacts/model.onnx` is STILL the old w16
  model - swapping the deliverable is the team's call.

### Done: `wide32` — GTCRN at width 32, from scratch

Trained 21 Sep 17:48 -> 22 Sep 12:25 IST, `configs/train_wide.yaml`. Stopped
deliberately at epoch 158/180: val PESQ had been flat at 1.920-1.922 for 25
epochs at LR < 1e-4. Best = epoch 132 (val PESQ 1.922 vs shipped 1.788).
Exported `artifacts/model_wide32{,_simple}.onnx` (103,381 params; streaming
matches offline, max abs diff 4.4e-7; `results/onnx_verify_wide32.json`).
Width 32 because it is the widest that keeps RTF < 0.5 on this laptop (w16
0.286, w32 0.482, w48 0.708 measured); final model: **RTF 0.484, p95 8.8 ms,
max 10.9 ms per 16 ms frame** (`results/bench_edge_wide32.json`, idle machine).

**Result — a defence specialist that lost generality:**

| | frozen defence set (720) PESQ-WB / STOI / out SNR | VoiceBank-DEMAND (824) PESQ-WB / STOI / out SNR |
|---|---|---|
| unprocessed | 1.319 / 0.801 / 4.41 | 1.968 / 0.921 / 8.45 |
| shipped (w16, fine-tuned from DNS3) | 1.931 / 0.860 / 10.79 | **2.387 / 0.921 / 16.34** |
| **wide32 (from scratch)** | **2.057 / 0.873 / 11.44** | 2.111 / 0.903 / 15.91 |

- Defence set: wide32 beats shipped on every metric and every category
  (paired: PESQ +0.125 better on 82% of clips, STOI +0.013 / 87%, SNR +0.65 dB
  / 83%, all p < 1e-58); burst SI-SDR gain gunshot 7.16 -> 8.12, artillery
  7.90 -> 8.44 dB. Passes all three targets at input SNR >= 10 dB.
- VoiceBank-DEMAND: wide32 is WORSE than shipped (PESQ -0.28, STOI -0.018,
  STOI now below unprocessed). Trained from scratch on LibriSpeech + defence
  noise only, it never saw what DNS3 pretraining gave shipped. Same asymmetry
  as the gtcrn_vctk finding in CLAUDE.md, in the other direction.
- `checkpoints/shipped_best.pt` / `artifacts/model.onnx` are UNCHANGED - which
  model to ship is an open decision. Next lever if time allows: fine-tune
  wide32 on mixtures that include general (non-defence) noise, or initialise a
  wide model from DNS3 weights (not possible across widths directly).
- `C:\SIH26052_data\STOP_TRAINING` was left in place so the .ps1 cannot
  restart the finished run by accident. Delete it before any new run.

Two things that cost time on the first attempt, both fixed:
- **Reading `C:` from WSL runs ~1.1 s/step; WSL-local disk runs 0.39 s/step.**
  The 19.7 GB of train/val audio the manifest references is mirrored at
  `/root/sih/data` (`C:\SIH26052_data\copy_to_wsl.sh`, list in
  `train_val_files.txt`) and selected with `SIH_DATA_ROOT=/root/sih/data`,
  which `train_wide32.sh` exports. The test set is still read from `C:`.
- **This laptop uses Modern Standby and slept for 2 h mid-copy.** The launcher
  now also asserts `ES_DISPLAY_REQUIRED` (display stays on). Lid-close or a
  manual Sleep still suspends it: keep it plugged in, lid open, for the run. New code behind it:
`src/models/gtcrn_wide.py` (subclasses upstream, overrides `__init__` only;
width 16 is bit-identical to upstream — `tests/test_wide.py`), width-aware
`train.py --width/--lr/--init`, `export_onnx.py`, `bench.py`, and cache shapes
read from the ONNX graph in `StreamingEnhancer`, `bench_edge.py` and the
handoff `SPEC.md`/`example_inference.py`.

- log: `C:\SIH26052_data\train_wide32.log`; per-epoch CSV: `results/train_log_wide32.csv`
- the shipped model scored **val_pesq ~1.79** on the same val set — the bar to beat
- launched detached by `C:\SIH26052_data\run_training_keep_awake.ps1`, which
  re-asserts "stay awake, display on" every 60 s and runs
  `train_wide32_loop.sh`: up to 30 attempts, each with `--resume`, so a crash
  costs at most the epoch in progress. Lid-close/Sleep PAUSES training (it
  survived a 24 min standby on 21 Sep and carried on).
- **phone alerts:** `C:\SIH26052_data\phone_updates.sh` runs detached in WSL
  and posts to ntfy.sh (topic in `C:\SIH26052_data\ntfy_topic.txt` - keep it
  out of git): crash/stop/stall/finish at once, PESQ milestones, 2-hourly
  status (07:00-24:00 only). Stop: `wsl -d Ubuntu-24.04 -u root -- pkill -f phone_updates.sh`.
- **stop on purpose:** create `C:\SIH26052_data\STOP_TRAINING`, then
  `wsl -d Ubuntu-24.04 -u root -- pkill -f src.train` (without the file, the
  loop restarts it). Delete the file before relaunching.
- **if the laptop was shut down / nothing is running:** re-run the .ps1
  (right-click > Run with PowerShell; double-click only opens it in Notepad).
- **Power setting changed for this run (21 Sep, at the owner's request):**
  sleep-when-plugged-in set to Never (was 15 min). Battery setting untouched.
  Undo after training: `powercfg /change standby-timeout-ac 15`.
- **if running late:** stop it, then relaunch with fewer epochs by adding
  `--epochs N` to the train line in `train_wide32_loop.sh` — the cosine LR is
  recomputed from the new total on resume, so no progress is lost.
- when done (or to check a mid-run best):

```bash
# in WSL: export + verify streaming ONNX of the best checkpoint
bash /mnt/c/SIH26052_data/wsl_run.sh -m src.export_onnx --ckpt checkpoints/wide32_best.pt --out artifacts/model_wide32.onnx
# on Windows: frozen-set eval, same instrument as the table above
python -m src.evaluate --workers 11 --tag onnx --methods onnx:artifacts/model_wide32_simple.onnx
```

It beat `shipped` on the frozen set but not on VoiceBank-DEMAND (see table
above), so the swap was left for the team to decide.

---

## Read this first: the project's status changed

The deliverable is built. **It does not yet do its job.**

Every artefact the problem statement asks for exists and is measured — model,
ONNX, spec sheet, results table, ablation, external calibration. On the frozen
synthetic test set the numbers are respectable. On **real recordings made with a
real microphone and real gunfire, the model makes speech HARDER to understand
than doing nothing at all.**

Measured with a speech recogniser standing in for a listener (whisper-medium,
26 known tokens per recording — see `scripts/asr_score.py`):

| recording | input SNR | unprocessed | model + floor −18 dB | model, full depth |
|---|---|---|---|---|
| `voice_noisy.wav`  (take 1) | +8 dB | **85%** | 73% | 12% |
| `voice_noisy2.wav` (take 2) | −12 dB | **50%** | 4% | 4% |
| `voice_noisy3.wav` (take 3) | −5.9 dB | **85%** | 77% | 69% |

Doing nothing wins on all three, and the ordering is monotonic: **the more the
model suppresses, the fewer words survive.** The gunfire is removed and the
speech goes with it.

That is the problem to solve. Everything below is organised around it.

### Why this was not caught earlier

The frozen test set says the model is good, because it measures PESQ, STOI and
SI-SDR against a clean reference on synthetic mixtures at **positive** SNR. None
of those is a measure of whether a listener can make out the words, and the test
set's SNR range does not cover the deployment case. Three separate proxy metrics
(STOI against a degraded reference, consonant-to-vowel ratio, band energy share)
all indicated improvement while word recognition fell. See
"Negative results" below — several plausible fixes were built, measured, and
found harmful.

### One large caveat, stated honestly

ASR is not an ear. Recognisers are trained on enormous quantities of noisy speech
and are far more robust to additive noise than humans, while being *less* robust
to processing artefacts they have never seen. Enhancement hurting ASR while
helping humans is a known effect. So the table above proves the chain hurts
**Whisper**; it does not by itself prove it hurts a person.

What makes it credible anyway: the human who recorded these files independently
reported the same thing repeatedly — gunfire gone, words unintelligible. Two
independent signals agreeing is much stronger than either alone.

**A scored human listening test is still the missing measurement.** The kit is
ready at `test-result/listening_test/` (answer sheet, key, scoring bands). It
needs three people who have not seen the script.

---

## What is running / nothing is running

No background jobs. GPU idle. Repo is in a clean, working state:
`19 passed` on the test suite.

---

## Where to pick up, in priority order

### 1. Decide whether the model belongs in the signal path at all

This is the honest first question, not a defeatist one. At +8 dB the full model
scores 12% against 85% unprocessed. If that reproduces under a human listening
test, then shipping it as-is is worse than shipping nothing, and the correct
engineering answer is either a better enhancer or a much shallower one.

The cheapest resolution is the listening test above.

### 2. Push the low-SNR training further

The one durable gain of the last session. The training mixture SNR was
`[0, +20] dB` — the steady background was **always quieter than the voice**, so
the model never saw the case the product exists for. `configs/data_lowsnr.yaml`
lowers it to `[-12, +12]`, verified by rendering actual mixtures:

| config | median realised SNR | below 0 dB | below −5 dB |
|---|---|---|---|
| `data.yaml` | +5.3 dB | 21% | 9% |
| `data_lowsnr.yaml` | **−3.8 dB** | **62%** | **41%** |

That produced `checkpoints/lowsnr_best.pt`, which beats the shipped model
substantially on real audio (take 3: 69% vs the shipped model's far worse
performance at the same depth). Next steps:

```bash
PY=.venv/bin/python

# push lower still
$PY -m src.train --tag lowsnr18 --data-config configs/data_lowsnr.yaml \
    --w-transient 0.0 --w-consonant 1.0     # then edit snr_db to [-18, 6]

# isolate the consonant term - the lowsnr run changed TWO things at once
$PY -m src.train --tag lowsnr_noconsonant --data-config configs/data_lowsnr.yaml \
    --w-transient 0.0 --w-consonant 0.0
```

**The `lowsnr` run changed both the SNR range and the loss, so attribution
between them is unknown.** The second command above settles it.

### 3. Stop selecting models on val PESQ

`checkpoint.monitor: val_pesq` in `configs/train.yaml`. The `lowsnr` run reached
its best val PESQ at **epoch 5** and then flatlined for 12 epochs until early
stopping — while PESQ is demonstrably not tracking word recognition. Selecting on
an ASR word score would optimise the thing that matters. This is a real change to
`src/train.py` and needs care: ASR scoring is slow, so it cannot run every epoch.

### 4. Understand why deep suppression destroys words — ANSWERED

`scripts/spectrogram_diff.py`, run on take 3's `floor_18dB.wav` vs
`floor_full_model.wav` (sample-aligned to each other — same input, same
pipeline, no timing correction needed). The original hypothesis here was
"over-gating brief high-frequency events" — **that is not what the data
shows.**

Measured, using this project's own CVR bands (`scripts/intelligibility.py`:
vowel 200–800 Hz, fricative/stop 2–6 kHz):

| | CVR |
|---|---|
| clean-speech reference | −10.68 dB |
| floor-capped (−18 dB) | −12.77 dB |
| full model | **−23.79 dB** |

Going from floor-capped to full model, the fricative band loses a median
**21.2 dB** more than it already had, against **11.1 dB** more in the vowel
band — the model disproportionately attacks the exact band that carries
consonant identity, as expected. What was NOT expected: the excess-kurtosis
of that extra suppression across time is **−0.59**, i.e. close to zero /
slightly *below* Gaussian — meaning it is **not concentrated in a few loud
transient frames**. The spectrogram diff plot
(`results/spectrogram_diff.png`) shows why directly: a near-continuous
suppression band from roughly 1–7 kHz runs through almost the ENTIRE 62 s
clip, not just around gunshots. **The full-depth model is not selectively
gating loud events — it is applying a broad, near-constant, aggressive
high-frequency rolloff for the whole recording,** and word loss is the
predictable result of doing that to a band speech identity lives in. This
also explains why the floor cap works as well as it does: capping
suppression DEPTH uniformly is a reasonably well-matched fix for a
uniformly-applied problem, not a event-detection problem.

**Implication for item 1 (training objective):** this is not a "the model
needs to detect transients better" problem — the transient detection isn't
the mechanism. It is producing an over-aggressive mask across the whole
signal, all the time, in exactly the band that matters. An ASR/word-loss
term in training should therefore penalize broadband high-frequency
suppression generally, not specifically penalize behavior at burst
boundaries.

### 5. The two older decisions, still open

- **Latency: 38.01 ms against a 32 ms target.** Unchanged and unchangeable with
  512/256 framing — 32 ms is the floor before any arithmetic. Fix is a 320/160
  retrain from scratch (~21 ms total). Now more attractive than before, because a
  from-scratch retrain is on the table anyway.
- **Artillery (66 clips) and rotor (72) rest on thin evidence.**
  `scripts/download_fsd50k_eval.sh` fills the gap, implies a full retrain, and
  `make_testset.py --force` invalidates every reported number.

---

## Negative results — do not rebuild these

Each was built as a plausible fix, measured, and found harmful or void. They are
kept (with warnings in their docstrings) because the measurements are the result.

| what | verdict |
|---|---|
| `scripts/post_enhance.py` | **Deleted.** Built on the belief the model was over-suppressing speech that was present. On the recording it was written for, the speech above 1 kHz genuinely was not there — it re-admitted noise, not voice. Premise disproved; do not rebuild a mask-floor post-processor without first checking the speech is actually in the band. |
| `scripts/voice_eq.py` — presence EQ toward a reference spectrum | Sounds better, measures worse. Costs 1.6 dB of consonant-to-vowel ratio; word score 58% → 42%. |
| `scripts/intelligibility.py` — consonant boost | Restores CVR to clean-speech parity (−16.8 → −10.9 dB) and still lowers word score. CVR is not intelligibility. |
| Multiband upward compression | The textbook move, and wrong here. Lifts every quiet frame, and most quiet frames are pauses and vowel tails rather than consonants: CVR −19.65 → −22.92 dB. Removed from `intelligibility.py`; do not reintroduce without measuring CVR. |
| Transient-weighted loss (earlier session) | No effect where intended (+0.05 dB on gunshot bursts, p = 0.72), significant cost elsewhere (PESQ −0.027, p < 0.001). `--w-transient` retained so the ablation is repeatable. |
| Dynamic INT8 quantization of the ONNX model | 5.5% *slower* on this CPU (compute here is ONNX Runtime dispatch-bound across 445 graph nodes, not arithmetic-bound — quantizing adds dequant/quant nodes rather than removing work) and 7.3% RMS output error vs fp32. Not shipped; see `scripts/bench_edge.py`. |
| NLMS/LMS/RLS reference-mic adaptive noise cancellation (`src/baselines/{nlms,lms,rls}.py`) | Every one of them makes real audio **worse than doing nothing**, and the ranking is the OPPOSITE of algorithmic sophistication: RLS (fastest, most complete convergence) is catastrophic, NLMS is bad, the "worst" algorithm — plain LMS with a conservative fixed step — does the *least* damage, purely because it adapts too slowly to fully exploit the problem below. Root cause: the (synthetic) reference-mic channel leaks some of the talker's own speech, and every one of these filters cannot distinguish "correlated because noise" from "correlated because leaked speech" — the more thoroughly an algorithm converges, the more speech it also removes. VAD-gated adaptation (the standard real-headset fix) does not help either in this regime: an energy-based VAD on the primary mic can't tell speech from noise when the noise is this loud and impulsive, so it freezes adaptation almost entirely and the result is indistinguishable from doing nothing. **Confirmed by real ASR word-recognition, not just PESQ/STOI** (whisper-medium, 26 known tokens, `results/asr_multimic.json`): unprocessed 62%, LMS ties it at 62%, NLMS drops to 54%, **RLS scores 0% — total destruction, every single word lost.** |
| RNNoise / DeepFilterNet as drop-in replacements | Both are real, working, pretrained-weight integrations (not stubs — see `src/methods.py`), and both **also** lose to unprocessed on this project's audio (SI-SDR gain −32.8 dB and −9.4 dB respectively on the same clip NLMS/LMS/RLS were measured on). The checkpoint that wins on generic noise keeps losing to doing nothing on gunfire — this is now confirmed by four independent published/pretrained systems (RNNoise, DeepFilterNet, `gtcrn_vctk`, and classical Wiener/spectral-subtraction), not just this project's own model. |
| NLMS + this project's own GTCRN in series (the "hybrid" architecture the DSP-frontend idea points toward) | ASR word score **4%**, worse than either failure alone (NLMS 54%, GTCRN-alone 54%) — stacking two degradations compounds rather than cancels. The GTCRN model alone, single-mic, on this same clip: ASR score **54%**, again below unprocessed's 62%, consistent with the project's central finding on a fourth independent recording. |

All of the above are measured on ONE synthetic 45 s clip (`results/multimic_demo/`,
real dry speech + real gunfire, not the frozen 720-clip testset) — see
`results/method_comparison.md` for the full consolidated table (now including the
ASR column) and `results/pareto_latency_quality.png` for the compute-cost-vs-PESQ
plot. Single data point per method, but the direction (everything loses to
unprocessed, confirmed by ASR not just PESQ/STOI) is now consistent across ten
different methods spanning classical DSP, adaptive filtering, and three
independent neural architectures.

---

## Measurement traps found the hard way

The instruments lie in specific, reproducible ways. All are now guarded in code.

1. **Whisper's temperature fallback is non-deterministic.** Left at its default
   tuple, a decode that trips an internal quality check is silently retried with
   *sampling*. The same unchanged file scored **69% and then 23%**. Fixed by
   passing `temperature=0.0` as a scalar. Always sanity-check with `--repeats 3`.
2. **Whisper drops or loops on hard audio.** Two distinct failures, both scoring
   ~4% on perfectly usable audio: a repetition loop (`"...A.M.A.M.A.M..."`), and
   silently skipping a segment — one file transcribed parts 1 and 4 correctly and
   omitted the alphabet and digits entirely, which is exactly the scored
   material. `looks_degenerate()` flags these now; **treat any flagged row as no
   measurement at all.**
3. **Chunked decoding is not a free fix.** It contains repetition loops but costs
   Whisper its context and destabilises other files — one file's *unprocessed*
   audio went from 69% to 0%. `--chunk` is opt-in; whole-file with a larger model
   is more reliable. Use `--model medium` for anything conclusive.
4. **STOI against a degraded reference is not intelligibility.** Scoring the
   output against the speaker's own muffled recording gave 0.84 and meant only
   "faithfully muffled".
5. **SI-SDR gain rises as SNR falls** (+6.3 dB at +10 dB SNR, +10.0 dB at −5 dB)
   because there is more noise to remove — it flatters exactly where
   intelligibility is collapsing.

---

## Recording quality: the capture chain matters more than expected

Two capture defects were found and fixed, both invisible until measured.

**Bluetooth headset mics are unusable.** In call mode they collapse to narrowband:
30 dB down by **1312 Hz**, against 4406 Hz for reference speech. Everything above
1.3 kHz — every consonant cue — was simply never recorded. No model recovers it.

**Phone VIDEO capture scoops the formant region.** Recording with the camera app
applies its own noise suppression and AGC. Switching to a plain **voice recorder**
app recovered **+19.2 dB at 1–2 kHz** and took consonant-to-vowel ratio from
−16.07 dB to −3.79 dB, at zero cost.

**For any future test recording:** voice recorder app, Bluetooth off, phone
10–15 cm from the mouth, gunfire from a speaker in the room (never through
earphones — the mic must hear it acoustically), and **leave ~10 s of gunfire
before speaking** so a clean noise bed can be extracted.

Verify any new recording before trusting a test built on it:

```bash
$PY scripts/asr_score.py --model medium --inputs your_dry_take.wav
```

A good dry take should score well above 70%. If it does not, fix the capture
before anything else.

---

## Environment: PyTorch availability

Torch 2.13.0+cu126 loads and trains normally on this machine, GPU included.

The torch-free inference split in `src/framing.py` remains correct and worth
keeping — it is what lets a teammate run the model without a 2 GB CUDA download —
but it is no longer load-bearing for *this* machine.

**The test suite reports `19 passed`.** It previously claimed "17 pass + 2 skip
without PyTorch"; in fact those two tests were broken — `tests/test_core.py`
bound `S` to `src.framing` (NumPy-only, no `stft`/`istft`) while two tests called
`S.stft`. They errored rather than skipping. Fixed.

---

## Files added in the last session

| path | what |
|---|---|
| `checkpoints/lowsnr_best.pt` | model trained on deployment-matched SNR |
| `artifacts/model_lowsnr_simple.onnx` | 390 KB streaming export, verified at 1.05e-06 |
| `configs/data_lowsnr.yaml` | mixture config with `snr_db: [-12, 12]` |
| `scripts/asr_score.py` | **the measurement rig** — word recognition, no listeners needed |
| `scripts/floor_sweep.py` | suppression-depth vs word-survival curve |
| `scripts/robustness_sweep.py` | operating envelope with real PESQ/STOI |
| `scripts/snr_sweep.py` | behaviour across input SNR on a given noise bed |
| `test-result/` | every real recording, every processed variant, all scores |
| `src/stream_demo.py` | gained `--floor-db` (live and file paths) |
| `src/losses.py` | gained `w_consonant` (1–4 kHz band term), default 0.0 |

## Files added in the Linux-migration / multi-mic session

| path | what |
|---|---|
| `main.py` | live mic→model→speaker entry point, wraps `src.stream_demo`, crash-proof fallback |
| `requirements.txt` | grouped, version-pinned dependencies (core/training/metrics/baselines) |
| `dhwanik.service`, `scripts/run_service.sh`, `dhwanik.env.example` | systemd **user** service — self-bootstraps `.venv`, installs deps only when `requirements.txt` changes, execs `main.py` |
| `scripts/bench_edge.py` | torch-free latency/RTF benchmark, safe to copy to an actual embedded target |
| `src/baselines/{nlms,lms,rls}.py`, `src/baselines/reference_mic.py` | reference-mic adaptive filters + synthetic second-channel model — see negative results above |
| `scripts/eval_multimic.py` | builds the synthetic two-mic mixture, runs all DSP/neural/hybrid methods, scores them |
| `scripts/_verify_new_baselines.py`, `src/methods.py` additions | RNNoise + DeepFilterNet, real pretrained weights (DeepFilterNet lives in an isolated `.venv-dfn` — numpy version conflict, see `requirements.txt`) |
| `scripts/aggregate_results.py` | consolidates every method measured on the same clip into `results/method_comparison.md` + `results/pareto_latency_quality.png` |
| `results/asr_multimic.json` | the ASR numbers in the table above |
| `scripts/make_listening_test.py`, `test-result/listening_test_v2/` | replacement listening-test kit — the original `test-result/listening_test/TEST_A.wav` has no recorded provenance anywhere in this repo (checked against every real recording and floor-sweep variant, no match), so a score against it can't be interpreted. This version uses the take-3 floor-sweep files (known provenance), independently randomized per listener. **Caveat:** those source files are also what README's own demo section asks teammates to listen to, so a listener who's used this repo already isn't blind. |
| `tests/test_nlms.py`, `tests/test_lms_rls.py`, `tests/test_streaming_safety.py` | 10 new tests, all passing (30/30 total, up from 19) |
| `scripts/spectrogram_diff.py`, `results/spectrogram_diff.png` | answers "where to pick up" item 4 below — the suppression mechanism |

---

## Rebuilding from nothing

```bash
# 1. environment - see README.md "Full setup from scratch"
# 2. data
bash scripts/download_tier1.sh
bash scripts/download_transients.sh

# 3. everything else, unattended and resumable
bash scripts/auto_pipeline.sh
```

`manifests/manifest.json` is not committed (22 MB of machine-specific absolute
paths); rebuild with `scripts/build_manifests.py`, about a minute. **Check the
group COUNTS it prints, not just that the assertions passed** — degenerate splits
are trivially disjoint. Healthy output has `background val=80`, `babble val=21`.

If you stop the pipeline mid-run, reap the workers:

```bash
pkill -f python
```

---

## Where things live

| what | where |
|---|---|
| code | this repository |
| venv | `.venv` at the repo root (Python 3.12, **not** 3.14) |
| datasets | `~/SIH26052_data/{raw,prepared}` (~64 GB) |
| frozen test set | `~/SIH26052_data/testset` |
| VoiceBank-DEMAND | `~/SIH26052_data/voicebank_demand` |
| real test recordings | `test-result/voice/` |
| processed variants + scores | `test-result/` |
