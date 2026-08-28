# Resume here

Current state, what is left, and exactly how to pick each item up.
Read `CLAUDE.md` for architecture and the invariants; `README.md` for results and
the full command reference.

---

## Status: the deliverable is complete

Everything the problem statement asks for exists and is measured. What remains is
optional improvement and two judgement calls.

| Item | State |
|---|---|
| Trained model | Done — `checkpoints/shipped_best.pt` |
| `model.onnx` for the hardware team | Done — 390 KB, self-contained, verified |
| Spec sheet + example inference code | Done — `artifacts/SPEC.md`, `example_inference.py` |
| Results sheet | Done — `results/results.md`, `results.csv`, `REPORT.md` |
| 60-second before/after recording | Done — `results/demo60/` |
| Baseline comparison (classical + pretrained) | Done |
| Ablation | Done — and it changed the conclusion, see below |
| External calibration (VoiceBank-DEMAND) | Done — reproduces published baseline exactly |
| Unit tests | 19 — all pass with PyTorch; 17 pass + 2 skip without it |
| Inference without PyTorch | Done — model, demos, baselines and tests all run |

---

## Blocker on this machine: Smart App Control

**Windows Smart App Control switched itself ON (enforcing) and now blocks
PyTorch's unsigned DLLs:**

```
OSError: [WinError 4551] An Application Control policy has blocked this file.
Error loading "...\torch\lib\c10.dll" or one of its dependencies.
```

Check with:
```powershell
(Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy" `
  -Name VerifiedAndReputablePolicyState).VerifiedAndReputablePolicyState
# 0 = off, 1 = ON (enforcing), 2 = evaluation
```

**What still works** — verified on this machine with torch unloadable:

| | |
|---|---|
| Shipped ONNX model on real audio | works, RTF 0.30 |
| `stream_demo` — file and live mic | works |
| Classical baselines + `evaluate.py` | works |
| PESQ / STOI metrics | work |
| Test suite | 17 pass, 2 skip with a clear reason |

This is deliberate: `src/framing.py` holds the constants with no torch
dependency, and `methods.py` imports torch lazily. See CLAUDE.md,
"The inference path must not import PyTorch".

**What is blocked:** training (`src/train.py`), ONNX export (`src/export_onnx.py`),
and the 2 torch-specific tests.

**To fix:** Windows Security → App & browser control → Smart App Control → Off.
Note this is **irreversible without a Windows reset** — Microsoft does not allow
turning it back on. If you only need to run and demo the model, you do not need
to touch it.

---

## Two decisions waiting on you

### 1. Latency: 38.01 ms measured against a 32 ms target

The floor is 32 ms *before any computation* — 16 ms to collect a chunk, plus a
measured 16 ms overlap-add delay (`win − hop`). No faster hardware changes this.

- **Option A — report it honestly.** Already written up in README and REPORT.md
  with the full breakdown. RTF passes comfortably (0.2955 vs < 0.5).
- **Option B — retrain at 320/160 framing.** 20 ms window → ~21 ms total, inside
  the target. Requires changing `src/stft.py` to `n_fft=320, hop=160, win=320`,
  adapting GTCRN's ERB/subband banding from 257 to 161 bins, and a full retrain
  (~2.2 h). The pretrained weights become unusable, so it trains from scratch —
  expect a lower absolute score.

### 2. Artillery and rotor rest on thin evidence

66 and 72 distinct source recordings, versus 2,165 for gunshot and 2,617 for
engine. Their scores are probably optimistic.

```powershell
bash scripts/download_fsd50k_eval.sh    # 6.2 GB, adds Explosion / Aircraft / more gunfire
# then re-run: prepare_data.py --step all, build_manifests.py,
# make_testset.py --force, and retrain
```

Note `make_testset.py --force` **invalidates every number already reported**, so
the whole results table must be regenerated together.

---

## What changed at the end, and why it matters

The ablation refuted the project's own design hypothesis.

The model was built with a loss term that weights gunfire frames harder — the
intuitive fix for "gunshots still get through". Retrained identically with that
term switched off and compared pairwise over the same 720 clips:

| effect of the transient term | result |
|---|---|
| Gunshot burst SI-SDR gain | +0.048 dB, p = 0.72 — no effect |
| Overall PESQ | −0.027, p < 0.001 — significantly worse |
| Overall STOI | −0.005, p < 0.001 — significantly worse |

So the shipped model is the version **without** it. The improvement over the
pretrained baseline is real (+1.376 dB on gunshot bursts, p = 0.0004) but comes
from **the training data, not the loss design**.

`--w-transient` is still wired up, so the experiment is repeatable:
```powershell
& $PY -m src.train --tag my_ablation --w-transient 0.0
```

---

## Optional next steps, roughly by value

1. **Regenerate the demo with a spectrogram figure.** The audio exists; a
   side-by-side spectrogram makes the gunshot removal visible as well as audible,
   which lands better in a slide deck than a table.
2. **Longer / better-tuned training.** The fine-tune improved validation PESQ by
   only +0.036 over 60 epochs, and the LR had decayed to ~0 by epoch 56. A higher
   peak LR or a restart schedule may extract more. The model is small (48K
   parameters) and may simply be capacity-limited.
3. **Indic-language speech.** The roadmap flags that judges notice English-only
   demos. AI4Bharat IndicVoices (gated on HuggingFace) or Common Voice Hindi
   would slot into `configs/data.yaml` as an additional speech pool; the pipeline
   needs no structural change.
4. **RNNoise as a third classical baseline.** Currently substituted by
   `noisereduce`. RNNoise runs at 48 kHz internally and needs 16→48→16
   resampling; it did not build cleanly on Windows.
5. **Drone/quadcopter audio.** Poorly covered by every open corpus surveyed.
   Would need a hand-curated Freesound pull or DroneAudioset.

---

## Rebuilding from nothing

The datasets (~64 GB) are not in the repo. To rebuild on a fresh machine:

```powershell
# 1. environment — see README.md "Full setup from scratch"
# 2. data
bash scripts/download_tier1.sh
bash scripts/download_transients.sh
# VoiceBank-DEMAND: use fetch_zip_until_valid.sh, not a size check (see CLAUDE.md)

# 3. everything else, unattended and resumable
bash scripts/auto_pipeline.sh
```

`auto_pipeline.sh` runs: extract → manifests → mixture QA → freeze test set →
**baseline table** → fine-tune → evaluate → export/bench/handoff/report → ablation.
Every stage skips completed work, and a failing stage aborts the chain rather
than letting later stages run on bad inputs.

**If you stop it mid-run**, kill the Python workers too — stopping the shell does
not always reap them:
```powershell
Get-Process python | Stop-Process -Force
```

Progress: `.\run.ps1 status`, or `tail -f C:\SIH26052_data\auto_pipeline.log`.

---

## Where things live

| what | where |
|---|---|
| code | this repository (`C:\dev\SIH-2026`) |
| venv | `C:\SIH26052_data\.venv` |
| datasets | `C:\SIH26052_data\{raw,prepared}` |
| frozen test set | `C:\SIH26052_data\testset` |
| VoiceBank-DEMAND | `C:\SIH26052_data\voicebank_demand` |
| pipeline log | `C:\SIH26052_data\auto_pipeline.log` |

Data and venv are outside the repo because 64 GB of audio and a 50,000-file
`site-packages` tree cannot go in git. `manifests/manifest.json` is likewise not
committed — 21 MB of machine-specific absolute paths, rebuilt by
`build_manifests.py` in about a minute.
