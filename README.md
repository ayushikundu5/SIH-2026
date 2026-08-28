# Smart Noise Cancellation for Defence Communication

**Smart India Hackathon 2026 — Problem Statement 26052 — AI/ML workstream**

A streaming speech-enhancement model that strips gunfire, artillery, rotor and
engine noise off a soldier's outgoing microphone feed. 48,245 parameters, small
enough for an embedded chip, fast enough to run live.

---

## The problem

A soldier speaks into a microphone while a rifle fires beside him and an engine
roars underneath. The radio carries all of it and the listener cannot make out
the words.

Conventional noise suppression assumes the background holds still — estimate the
noise floor, subtract it. A gunshot breaks that assumption completely: it arrives
in microseconds, peaks far above the speech, and is gone before any noise tracker
adapts. We measured exactly that failure:

| Method | Gunfire removed (SI-SDR gain inside bursts) |
|---|---|
| Wiener filter | **+0.19 dB** — essentially nothing |
| Spectral subtraction | **+0.11 dB** — essentially nothing |
| GTCRN, off-the-shelf | +5.84 dB |
| **This project** | **+7.16 dB** |

Roughly **80% of gunshot noise energy removed, versus ~4% for classical filters.**

### Scope boundary

This repository is **Path 1 only**: the neural network that cleans the *outgoing
transmitted voice*. Earcup rumble cancellation, the DAC/driver chain, and the
analog limiter ahead of the ADC belong to the hardware team.

The model performs **suppression only — it does not apply gain.** Voice
amplification belongs to the analog path downstream.

**Contract with the hardware team:** 16 kHz mono in, 16 ms chunks, same out.

---

## Results

Measured on a **frozen 720-clip test set** — held-out speakers, held-out noise
recordings, held-out rooms — plus the standard VoiceBank-DEMAND benchmark.

### Per noise category (PESQ, whole clip)

| Method | Gunshot | Artillery | Rotor | Engine | Siren | Babble | Overall |
|---|---|---|---|---|---|---|---|
| **Ours (shipped)** | **1.901** | **1.722** | **1.995** | **2.066** | **2.080** | **1.833** | **1.933** |
| GTCRN pretrained (DNS3) | 1.793 | 1.628 | 1.868 | 1.967 | 1.934 | 1.702 | 1.815 |
| Wiener filter | 1.316 | 1.205 | 1.404 | 1.422 | 1.408 | 1.374 | 1.355 |
| Spectral subtraction | 1.303 | 1.197 | 1.372 | 1.395 | 1.375 | 1.352 | 1.332 |
| Unprocessed | 1.297 | 1.188 | 1.354 | 1.380 | 1.364 | 1.333 | 1.319 |
| GTCRN VoiceBank-tuned | 1.311 | 1.210 | 1.354 | 1.357 | 1.351 | 1.320 | 1.317 |

Note the last row: the checkpoint that **wins** on VoiceBank-DEMAND (PESQ 2.868
there, better than DNS3's 2.508) is statistically indistinguishable from doing
nothing here. The model that handles café noise best is the worst at gunfire —
the clearest possible argument for domain-specific training.

### Targets, by input-SNR band

Achievable gain is bounded by how much noise was present to begin with —
measured correlation between input SNR and SNR gain is **−0.78**. A single
average is therefore misleading in both directions, so results are stratified.

| Input SNR | Clips | Unprocessed | Ours | PESQ > 2.5 | STOI (ours) | STOI > 0.85 |
|---|---|---|---|---|---|---|
| < 0 dB | 170 | 1.098 | 1.378 | No | 0.733 | No |
| 0 – 5 dB | 232 | 1.176 | 1.723 | No | 0.852 | **Yes** |
| 5 – 10 dB | 169 | 1.357 | 2.099 | No | 0.911 | **Yes** |
| > 10 dB | 149 | 1.752 | **2.706** | **Yes** | 0.957 | **Yes** |

### Speed and latency

| Stage | ms |
|---|---|
| Chunk buffering (hop) | 16.00 |
| Overlap-add delay (**measured**, = `win − hop`) | 16.00 |
| Model compute (p95, 1 thread) | 6.01 |
| **Total** | **38.01** |

**RTF 0.2955** (target < 0.5) — passes comfortably.
**Latency 38.01 ms** against a 32 ms target — **does not pass, and cannot.**
See [Known limitations](#known-limitations).

### Is the measurement trustworthy?

The same evaluator was run over the standard VoiceBank-DEMAND test set, where
published numbers exist to check against:

| | Ours, measured | Published |
|---|---|---|
| Unprocessed PESQ | **1.968** | 1.97 |
| Unprocessed STOI | **0.921** | 0.921 |

The pipeline reproduces the published baseline exactly, which independently
verifies the resampling chain, the ITU-T P.862 PESQ build, the STFT framing and
the inference path. Every other number here rests on that calibration.

---

## For teammates — start here

There are two very different levels of setup. Pick the one you need.

### Level 1 — run and demo the model (5 minutes, no dataset)

**You do not need the 64 GB of audio.** The trained model is committed to this
repo. This is enough to hear it work, demo it, and run it on your own recordings.

```powershell
git clone https://github.com/ayushikundu5/SIH-2026.git
cd SIH-2026

# any Python 3.9-3.12; no GPU, no PyTorch
pip install numpy soundfile soxr onnxruntime sounddevice pyyaml

# listen to the 60-second demo that is already in the repo
cd results\demo60
start before.wav
start after.wav
```

Then run it on your own audio, or live from your microphone:

```powershell
python -m src.stream_demo --file yourfile.wav --onnx artifacts\model_simple.onnx
python -m src.stream_demo --live      # WEAR HEADPHONES or it feeds back
```

### Level 2 — retrain or re-evaluate (hours, needs the dataset)

Only needed if you are changing the model or regenerating results. See
[Full setup from scratch](#full-setup-from-scratch) and
[Reproducing everything](#reproducing-everything).

**Use the same absolute paths as the original machine and nothing needs editing:**

```
C:\SIH26052_data\.venv     the virtualenv
C:\SIH26052_data\raw       downloaded corpora
C:\SIH26052_data\prepared  16 kHz conversions
C:\SIH26052_data\testset   frozen evaluation set
```

If you use different paths, edit the four lines that reference them:
`configs/data.yaml` (3 lines: `raw`, `prepared`, `testset`) and `run.ps1`
(1 line: `$PY`).

Downloads total **~32 GB** and take several hours. They are resumable — re-run
the same script if interrupted, and it continues rather than restarting.

---

## Quick start — test it yourself

The shipped model runs through **ONNX Runtime only**. No GPU, no PyTorch needed.

### 1. Listen to the 60-second demo

```powershell
cd results\demo60
start before.wav      # voice + engine + 10 gunshots
start after.wav       # same clip, cleaned
```

Gunshots occur at **20.4, 25.5, 29.9, 32.9, 35.2, 39.6, 44.4, 48.3, 53.4, 56.0 s**.
The clearest improvements are at 25 s, 33 s and 35 s. Use headphones.

Measured on that clip: PESQ **1.40 → 2.19**, STOI **0.91 → 0.95**,
SI-SDR **6.3 → 13.0 dB**.

### 2. Run it on your own audio

```powershell
$PY = "C:\SIH26052_data\.venv\Scripts\python.exe"
& $PY -m src.stream_demo --file path\to\your.wav --onnx artifacts\model_simple.onnx
# writes results\demo\before.wav and results\demo\after.wav
```

### 3. Live microphone

```powershell
& $PY -m src.stream_demo --list     # list audio devices
& $PY -m src.stream_demo --live     # speak, clap, bang the desk; Ctrl+C to stop
```

**Wear headphones** or the microphone will pick up the speakers and feed back.
This runs the real streaming contract — one 16 ms frame per callback, caches
carried forward — so it is the same code path the hardware team will run.

### 4. Regenerate the demo

```powershell
& $PY scripts\make_demo.py           # rebuilds results\demo60\ from held-out data
```

---

## Full setup from scratch

### Prerequisites

```powershell
# Python 3.12 — NOT 3.14. PyTorch ships CPU-only wheels for 3.14, so on 3.14 the
# GPU sits idle and training is roughly 50x slower with no error message.
winget install --id Python.Python.3.12 --scope user

# Virtualenv, deliberately OUTSIDE any cloud-synced folder
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv C:\SIH26052_data\.venv
$PY = "C:\SIH26052_data\.venv\Scripts\python.exe"

# PyTorch with CUDA. Only needed for TRAINING and EXPORT — not for running the model.
& $PY -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126

& $PY -m pip install numpy scipy soundfile soxr librosa pystoi onnx onnxruntime `
                     onnxscript onnxsim einops sounddevice pyyaml tqdm matplotlib `
                     pandas pytest
```

Verify CUDA: `& $PY -c "import torch; print(torch.cuda.is_available())"` → must print `True`.

### PESQ needs a C compiler

`pesq` is the ITU-T P.862 reference implementation and has no cp312 Windows
wheel, so it builds from source:

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools `
  --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
& $PY -m pip install pesq
```

If that still fails with *"Microsoft Visual C++ 14.0 or greater is required"*
even after the build tools install, `vswhere` has not registered the toolchain.
Build against the environment directly:

```powershell
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "call `"$vcvars`" && set DISTUTILS_USE_SDK=1 && set MSSdk=1 && `"$PY`" -m pip install pesq"
```

### Vendored upstream code

`third_party/gtcrn` holds the streaming-conversion code from
[Xiaobin-Rong/gtcrn](https://github.com/Xiaobin-Rong/gtcrn) (MIT), which
`src/export_onnx.py` imports. The model itself is vendored at
`src/models/gtcrn.py` with its licence alongside.

---

## Reproducing everything

Datasets live at `C:\SIH26052_data` — **~64 GB**, deliberately outside the repo.
They are fully reproducible from the scripts here.

```powershell
bash scripts/download_tier1.sh          # LibriSpeech, MUSAN, RIRs, ESC-50   (~24 GB)
bash scripts/download_transients.sh     # firearm corpus, UrbanSound8K       (~8 GB)
```

VoiceBank-DEMAND needs the content-validating fetcher, because
`datashare.ed.ac.uk` answers HEAD with a ~4 KB HTML interstitial and a
size-based check will happily declare a 10%-downloaded file complete:

```powershell
bash scripts/fetch_zip_until_valid.sh `
  "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip" `
  /c/SIH26052_data/raw/vbd_clean_testset.zip
bash scripts/fetch_zip_until_valid.sh `
  "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip" `
  /c/SIH26052_data/raw/vbd_noisy_testset.zip
```

Then the whole pipeline — resumable, skips completed work, aborts rather than
running a later stage on bad inputs:

```powershell
bash scripts/auto_pipeline.sh
```

Or stage by stage:

```powershell
.\run.ps1 status      # what is downloaded / built so far
.\run.ps1 data        # extract + resample + manifests + mixture QA
.\run.ps1 testset     # freeze evaluation set + VoiceBank-DEMAND benchmark
.\run.ps1 baseline    # comparison table — RUN BEFORE TRAINING
.\run.ps1 train       # fine-tune (~2.2 h on an RTX 3050)
.\run.ps1 ablate      # identical run, transient loss term disabled
.\run.ps1 finish      # evaluate + bench + export + handoff bundle
.\run.ps1 test        # unit tests
```

---

## Testing

### Unit tests — 19 invariants

```powershell
$PY = "C:\SIH26052_data\.venv\Scripts\python.exe"

& $PY -m pytest tests -q                                              # all
& $PY -m pytest tests -q -v                                           # verbose
& $PY -m pytest tests/test_core.py::test_wola_roundtrip_is_exact -q   # single test
& $PY -m pytest tests -q -k mixer                                     # by keyword
& $PY -m pytest tests -q -x                                           # stop at first failure
```

These pin down the things that **fail silently** rather than raising: STFT
round-trip exactness, transient-mask/frame alignment, that the mixer actually
achieves the SNR it was asked for, that the limiter stays monotonic and bounded,
and that SI-SDR is genuinely scale-invariant.

### Fast end-to-end checks (each under a minute)

```powershell
# data → loss → backward, with real data; catches shape/device/NaN problems
& $PY scripts\smoke_train.py --steps 20 --batch 24 --workers 8

# the full trainer, two tiny epochs — exercises scheduler, validation, checkpointing
& $PY -m src.train --tag smoke --epochs 2 --epoch-size 480 --val-size 96

# render mixtures AND check them (mask coverage + burst prominence)
& $PY scripts\qa_mixtures.py --n 24

# evaluate a few methods on a subset
& $PY -m src.evaluate --methods unprocessed wiener gtcrn_dns3 --limit 40

# latency + RTF — run on an IDLE machine, background load inflates it
& $PY -m src.bench --onnx artifacts\model_simple.onnx

# verify the shipped ONNX still matches the offline model
& $PY -m src.export_onnx --ckpt checkpoints\shipped_best.pt --out artifacts\model.onnx
```

### Evaluating a specific checkpoint

```powershell
& $PY -m src.evaluate --methods "gtcrn:checkpoints/shipped_best.pt" --tag mine
& $PY -m src.evaluate --testset C:\SIH26052_data\voicebank_demand `
      --methods unprocessed gtcrn_vctk --tag vbd
```

Method names accepted: `unprocessed`, `wiener`, `specsub`, `noisereduce`,
`gtcrn_dns3`, `gtcrn_vctk`, or `gtcrn:<path-to-checkpoint>`.

---

## Datasets

| Purpose | Dataset | Licence |
|---|---|---|
| Clean speech | LibriSpeech `train-clean-100` / `dev-clean` / `test-clean` | CC BY 4.0 |
| Steady noise | MUSAN (noise, music, speech-for-babble) | CC BY 4.0 |
| Room acoustics | OpenSLR-28 impulse responses | Apache 2.0 |
| **Gunfire** | Zenodo 7004819 — 4 firearms × 3 firing styles, outdoor range, AFRL-affiliated | **CC BY 4.0** |
| Transients | UrbanSound8K (`gun_shot`, siren, engine) | CC BY-NC 4.0 |
| Transients | ESC-50 (helicopter, fireworks, thunderstorm, engine, siren) | CC BY-NC 3.0 |
| Benchmark | VoiceBank-DEMAND test set | see record EULA |

The firearm corpus ships **ground-truth gunshot timestamps** — 6,212 annotated
shots across 2,148 clips — so bursts are cut at the real acoustic event rather
than wherever an energy detector guesses.

> **ESC-50 has no `gun_shot` class.** It offers `fireworks` and `thunderstorm` as
> transient proxies only. Real firearm audio comes from Zenodo 7004819 and
> UrbanSound8K.

Pool sizes after splitting (train / val / test, disjoint by recording):

| Pool | Train | Val | Test |
|---|---|---|---|
| Speech (LibriSpeech) | 27,269 | 1,940 | 1,850 |
| Gunshot | 2,165 | 132 | 265 |
| Engine | 2,617 | 188 | 315 |
| Siren | 1,122 | 41 | 275 |
| Babble | 362 | 21 | 43 |
| Artillery | 66 | 4 | 10 |
| Rotor | 72 | 2 | 6 |
| Room impulse responses | 51,112 | 3,101 | 6,005 |

---

## How it works

```
download_*.sh      → raw corpora (resumable; these hosts drop long transfers)
prepare_data.py    → extract, resample ONLY what needs it
build_manifests.py → splits + disjointness ASSERTIONS → manifests/manifest.json
qa_mixtures.py     → render mixtures and CHECK them
make_testset.py    → frozen evaluation set, written once
src/evaluate.py    → comparison table  (BEFORE training)
src/train.py       → fine-tune
src/bench.py       → latency + RTF
src/export_onnx.py → streaming ONNX + faithfulness check
make_handoff.py    → the bundle the hardware team receives
make_report.py     → assembles results/REPORT.md from measured files only
```

### The mixer is the centre of gravity

Mixtures are built in **two distinct layers**, and that distinction is the whole
project:

- **Steady background** (MUSAN) at a global SNR measured over *active-speech
  frames only* — whole-clip SNR is skewed by silence, so two clips nominally at
  "0 dB" sound quite different.
- **Impulsive bursts** (gunfire, explosions) placed at random offsets as discrete
  events, cut at their acoustic onset, scaled by **peak** level, and routinely
  louder than the speech.

Blending gunfire evenly through a clip instead of injecting discrete bursts
teaches only the easy stationary case, and the model then fails on exactly the
sound the problem statement cares about.

The mixer also models the **analog limiter** the hardware team places before the
ADC (soft-knee, threshold 0.6–0.95), because that is the signal the model
actually receives in the field.

### Burst-local metrics

Gunfire occupies 12–20% of a clip. A whole-clip score is dominated by the
remainder, so a model that removes *none* of the gunfire can still post a
respectable number. `metrics.masked_metrics` measures **inside the bursts** — the
column that answers the question actually being asked.

### Splits are disjoint by RECORDING, and asserted

`build_manifests.py` **raises** on overlap rather than warning. This matters more
than it sounds: the firearm corpus ships each shot as up to 8 per-channel files
plus a channel-mean file sharing one `uuid`, so a naive per-file split would leak
the same physical gunshot into train and test eightfold.

---

## What the hardware team receives

Everything in `artifacts/`:

- **`model.onnx`** — 390 KB, weights folded inline, verified self-contained by
  loading it from an empty scratch directory
- **`SPEC.md`** — sample rate, chunk size, cache shapes, latency breakdown, opset
- **`example_inference.py`** — minimal streaming loop
- **`results.md` / `results.csv`** — the measurements
- **`passthrough_stub.onnx`** — a Day-1 identity model with the real interface, so
  hardware integration could be built and tested while training ran

The model is **stateful and streaming**: one 16 ms frame per call, with three
cache tensors fed back in.

| Input | Shape |
|---|---|
| `mix` | `(1, 257, 1, 2)` — STFT of one frame, real/imag |
| `conv_cache` | `(2, 1, 16, 16, 33)` |
| `tra_cache` | `(2, 3, 1, 1, 16)` |
| `inter_cache` | `(2, 1, 33, 16)` |

Initialise all caches to zeros; feed the outputs back on the next call.

Streaming fidelity was verified frame-by-frame against the offline model:
**max absolute difference 3.99e-07**.

---

## Known limitations

Stated plainly, because a results table that hides these is worth less than one
that admits them.

1. **The 32 ms latency target cannot be met** with GTCRN's 512/256 framing.
   16 ms of chunk buffering plus a *measured* 16 ms overlap-add delay is 32 ms
   before a single multiply; the total is 38.01 ms. No faster processor fixes
   this. The fix is 320/160 framing (20 ms window, ~21 ms total) and a retrain.

2. **The transient-weighted loss term was tested and dropped.** It was built as
   the structural answer to "gunshots still get through", but a paired ablation
   over the same 720 clips showed **no benefit where it was supposed to help**
   (+0.05 dB on gunshot bursts, p = 0.72) and a small but **significant cost** to
   overall quality (PESQ −0.027, p < 0.001). The shipped model is the version
   *without* it. The improvement over the pretrained baseline is real
   (+1.38 dB on gunshot, p = 0.0004) but comes from **the training data, not the
   objective**. The code retains `--w-transient` so the experiment is repeatable.

3. **`SNR gain > 15 dB` is not well-posed on its own** (measured correlation with
   input SNR: −0.78). Reported by band instead.

4. **Gunfire suppression is uneven.** The +7 dB figure is an average; individual
   shots range from clearly removed to barely attenuated. This is "substantially
   reduces gunfire", not "removes it".

5. **Artillery and rotor rest on thin evidence** — 66 and 72 distinct source
   recordings, against 2,165 for gunshot and 2,617 for engine.
   `scripts/download_fsd50k_eval.sh` fills the gap but implies a retrain.

6. **Drone/quadcopter audio is poorly covered** by open corpora. Helicopter is
   well represented; drones are not.

7. **ESC-50 and UrbanSound8K are CC BY-NC.** Fine for research and competition;
   must be replaced with CC-BY sources before procurement. The firearm corpus is
   already CC BY 4.0.

8. **PyTorch exports at opset 18** regardless of the requested version — confirm
   your runtime supports it before any TensorRT/Jetson port.

---

## Repository layout

```
configs/        data.yaml (corpora, SNR ranges, burst params), train.yaml
src/
  audio.py      I/O, resampling, level maths, active-speech VAD
  stft.py       512/256 sqrt-Hann framing — must match GTCRN exactly
  mixer.py      ★ two-layer mixture synthesis + transient mask
  dataset.py    on-the-fly mixing (train), frozen set (val/test)
  losses.py     compressed-spectrum loss + optional transient weighting
  metrics.py    PESQ, STOI/ESTOI, SI-SDR, segSNR, burst-local variants
  methods.py    uniform registry so every method is swept identically
  baselines/    Wiener (decision-directed), spectral subtraction (Boll)
  models/       GTCRN, vendored verbatim (MIT)
  train.py  evaluate.py  bench.py  export_onnx.py  stream_demo.py
scripts/        download, prepare, manifests, QA, testset, demo, handoff, report
tests/          19 invariant tests
artifacts/      what the hardware team receives
results/        measured outputs, training logs, the 60 s demo
```

`manifests/` and the datasets are generated, not committed — see
[Reproducing everything](#reproducing-everything).

---

## Attribution

Base model: **GTCRN** — Rong et al., *"GTCRN: A Speech Enhancement Model
Requiring Ultralow Computational Resources"*, ICASSP 2024. 48,245 parameters
(measured), MIT licence. Vendored at `src/models/gtcrn.py`; streaming conversion
and ONNX export use the upstream `stream/` implementation.

Metrics use reference implementations only — `pesq` (ITU-T P.862) and `pystoi`.
No perceptual metric is hand-rolled.
