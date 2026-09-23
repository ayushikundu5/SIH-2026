# Training the wide48 model on a second machine (Linux)

**For:** the teammate lending a Linux desktop with an NVIDIA GTX 1660 Super.
**Time:** about 1 hour of your attention, then ~18–20 hours of the machine
training on its own. You can use the PC for other things meanwhile; it will
just be slower.
**You send back:** one file of about 2.3 MB.

Everything below is copy-pasteable. Where you must substitute something it is
written like `<this>`.

---

## 1. What this is, in one minute

Smart India Hackathon 2026, problem 26052 (DRDO): software that removes
battlefield noise — gunfire, artillery, helicopters, vehicles — from a
soldier's microphone so the voice stays intelligible over the radio. A small
neural network runs on the audio live, 16 ms at a time.

The problem statement sets three targets. Measured on our frozen 720-clip
defence test set, the current model stands at:

| Target | Required | Current model | |
|---|---|---|---|
| STOI (intelligibility) | > 0.85 | 0.876 | ✅ |
| PESQ (speech quality) | > 2.5 | 2.106 | 0.39 short |
| Output SNR | > 15 dB | 11.7 dB | 3.3 dB short |
| Real-time factor | < 0.5 | 0.476 | ✅ |

The model has grown 48k → 103k parameters so far, and every capacity increase
has raised quality (PESQ 1.931 → 2.057 → 2.106). **Your run is the next step:
190k parameters (width 48).** It answers whether more capacity keeps paying.

Two other things are happening in parallel on Ayushi's laptop, so please don't
worry about them: a clean retrain at the current size, and a two-microphone
version that uses the helmet's outer mic.

**Why your machine.** Training is one long GPU job. Ayushi's laptop can only
run one at a time, and each takes most of a day. Your card is about as fast as
hers and has the same 6 GB of memory, so running this one on yours roughly
halves the calendar time.

---

## 2. What you need before starting

```bash
nvidia-smi                 # must print a table with the GTX 1660 Super
df -h ~                    # need ~26 GB free
python3 --version          # any; we install 3.12 below if needed
free -g                    # 8 GB RAM or more is comfortable
nproc                      # number of CPU threads - note it, used in step 6
```

**Driver:** `nvidia-smi` prints a driver version top-right. It must be **525 or
newer** for the CUDA build of PyTorch. If it's older, update the driver first
(`sudo ubuntu-drivers autoinstall` on Ubuntu, then reboot).

**Disk budget:** audio data 20 GB + Python environment 5 GB + checkpoints and
logs under 1 GB.

---

## 3. Get the code

The repository is **private**, so first ask Ayushi to either add your GitHub
account as a collaborator, or send you a zip of it.

```bash
mkdir -p ~/sih && cd ~/sih
git clone https://github.com/ayushikundu5/SIH-2026.git
cd SIH-2026
```

If you got a zip instead:

```bash
mkdir -p ~/sih && cd ~/sih
unzip ~/Downloads/SIH-2026.zip -d .
cd SIH-2026
```

---

## 4. Install Python 3.12 and the system libraries

Ubuntu 24.04 already has Python 3.12. On 22.04 or older, add it:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
```

Then, on any version:

```bash
sudo apt install -y python3.12 python3.12-venv python3.12-dev \
                    build-essential libsndfile1 git
```

`libsndfile1` reads the audio files; `build-essential` and `python3.12-dev` are
needed because the PESQ scorer compiles a small C extension during install.

---

## 5. Create the Python environment

```bash
cd ~/sih/SIH-2026
python3.12 -m venv .venv
.venv/bin/pip install --no-cache-dir --upgrade pip wheel setuptools

# PyTorch with CUDA 12.6 - about 3 GB, this is the long one
.venv/bin/pip install --no-cache-dir torch==2.13.0 \
    --index-url https://download.pytorch.org/whl/cu126

# everything else
.venv/bin/pip install --no-cache-dir numpy==2.5.2 scipy==1.18.1 pandas \
    PyYAML==6.0.3 tqdm==4.70.0 soundfile==0.14.0 soxr==1.1.0 einops==0.8.2 \
    onnx==1.22.0 onnxruntime==1.29.0 onnxscript onnxsim pystoi==0.4.1 \
    pytest cython

# PESQ compiles against the numpy already installed, so it goes last
.venv/bin/pip install --no-cache-dir --no-build-isolation pesq==0.0.4
```

> **Do not run `pip install -r requirements.txt`.** That file pins the
> **CPU** build of PyTorch (it is used by teammates who only run the finished
> model), and it would replace the CUDA build you just installed.

Check the GPU is visible from Python:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.13.0+cu126 True NVIDIA GeForce GTX 1660 SUPER`.
If it prints `False`, the driver is too old — see step 2.

---

## 6. Get the audio data (~20 GB)

Ayushi exports exactly the files this run needs onto a pendrive or external
disk (see section 11 for her command). It arrives as a folder called
`sih_data` containing `raw/`, `prepared/` and `manifest_combat.json`.

```bash
# copy it off the drive (adjust the source path)
cp -r /media/$USER/<PENDRIVE>/sih_data ~/sih_data

# the manifest belongs in the repo
cp ~/sih_data/manifest_combat.json ~/sih/SIH-2026/manifests/

# check it arrived complete: expect about 92,858 files and 20 GB
find ~/sih_data -type f | wc -l
du -sh ~/sih_data
```

**Tell the code where the data is.** The manifest was written on Windows and
holds paths like `C:\SIH26052_data\raw\...`; this variable rewrites them to
your machine automatically, so nothing needs editing:

```bash
export SIH_DATA_ROOT=$HOME/sih_data
echo 'export SIH_DATA_ROOT=$HOME/sih_data' >> ~/.bashrc   # so it survives reboots
```

---

## 7. Verify the setup before the long run

Two checks. Both must pass, and together they take about three minutes.

```bash
cd ~/sih/SIH-2026

# 1. the test suite - 36 checks of the maths, framing and model wrappers
.venv/bin/python -m pytest tests -q
# expected: "36 passed"

# 2. a one-minute training smoke test: real data, real GPU, 20 steps
SIH_DATA_ROOT=$HOME/sih_data .venv/bin/python scripts/smoke_train.py \
    --steps 20 --batch 24 --workers 8 \
    --data-config configs/data_combat.yaml \
    --manifest manifests/manifest_combat.json
```

(The test suite needs no audio, so it also passes before the data arrives —
handy if you want to check the install while the copy runs.)

The smoke test should print a falling loss and finish without errors. **If it
fails with "no such file" the data path is wrong** — check `SIH_DATA_ROOT`
points at the folder that directly contains `raw/` and `prepared/`.

---

## 8. Stop the machine from sleeping

A suspend mid-run is survivable (training resumes on wake), but it wastes
hours. Disable sleep while you're lending the machine:

```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

Undo it afterwards with:

```bash
sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

Closing a laptop lid or the screen blanking is fine; the GPU keeps working.

---

## 9. Start the training

```bash
cd ~/sih/SIH-2026

# if `nproc` in step 2 showed fewer than 8, edit num_workers in
# configs/train_wide48.yaml to roughly your physical core count first

CFG=configs/train_wide48.yaml TAG=wide48 nohup bash scripts/run_training_loop.sh > /dev/null 2>&1 &
```

`nohup ... &` keeps it running after you close the terminal. The loop restarts
training by itself if it crashes, always resuming from the last finished epoch.

**Watch it:**

```bash
tail -f train_wide48.log          # live; Ctrl-C just stops watching
grep "^epoch" train_wide48.log | tail -5    # completed epochs
nvidia-smi                        # should show python using ~4.2 GB
```

A healthy log looks like this (your numbers will differ):

```
GTCRN width 48, parameters: 190,005   device: cuda
  e000 s0050/834  loss= 22.9108  lr=1.30e-05
epoch 000  train=15.7397  val=8.4532  val_pesq=1.318  val_stoi=0.733  (9.1 min)
```

**What to expect:** about **9 minutes per epoch, 120 epochs, so 18–20 hours.**
`val_pesq` starts near 1.3 and climbs past 2.0; it wobbles by ±0.03 between
epochs, which is normal.

**After a reboot or if you stopped it,** just run the same `CFG=... nohup ...`
line again. It continues from the last saved epoch.

**To stop it deliberately:**

```bash
touch STOP_wide48
pkill -f "src.train --config configs/train_wide48.yaml"
rm STOP_wide48        # delete this before you want to start again
```

---

## 10. When it finishes, send two files back

The log ends with `done. best val_pesq = <number>`. Then:

```bash
ls -la checkpoints/wide48_best.pt      # ~2.3 MB - this is the model
ls -la results/train_log_wide48.csv    # small - the learning curve
```

Send both to Ayushi (WhatsApp, Drive, anything). **Please don't delete the
data or the environment yet** — if the result looks promising we may ask for
one more run, and the setup is the slow part.

Evaluation happens on Ayushi's laptop, because it holds the frozen test sets.
That's deliberate: every number in the project comes from one machine and one
evaluator, so the comparisons stay honest.

---

## 11. What Ayushi does on her side (for context)

**Export the data for you** (once, onto a pendrive):

```powershell
python scripts/export_training_data.py --dest E:/sih_data --dry-run   # check size
python scripts/export_training_data.py --dest E:/sih_data             # ~20 GB
```

This copies only the files the training manifest actually names, in the
train/val splits — not the full 59 GB of corpora, and no test-split audio, so
the test sets cannot leak into your training.

**In parallel on her laptop:**
- **Phase 1** (`fresh32`): the same width-32 model retrained from scratch. Until
  22 Sep a bug meant every epoch reused one fixed set of 20,000 mixtures, so
  every earlier model, including the current best, saw far less variety than
  intended. This is the first clean run.
- **Phase 1b:** a version with shorter audio frames, to get total delay under
  the 32 ms target (currently 41 ms).
- **Phase 3:** a two-microphone model. The helmet has a mic inside at the mouth
  and one on top that mostly hears the surroundings. The second mic gives the
  model a direct measurement of the noise, which is the realistic route to the
  15 dB SNR target.

**When your run lands**, she quantises it to 8-bit (the problem statement asks
for this) and measures whether width 48 fits the real-time budget after
compression. Width 48 is currently too slow on a laptop CPU (RTF 0.708 against
the 0.5 target), which is exactly what the quantisation step is meant to fix.

---

## 12. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `torch.cuda.is_available()` is `False` | Driver older than 525. `sudo ubuntu-drivers autoinstall`, reboot. |
| `CUDA out of memory` | Something else is using the GPU, or the card has less free memory than expected. Set `batch_size: 12` in `configs/train_wide48.yaml` and restart; the run slows a little but is otherwise unaffected. |
| `pesq` fails to install | `python3.12-dev` or `build-essential` missing, or numpy was not installed first. Re-run step 4, then the last pip line with `--no-build-isolation`. |
| `FileNotFoundError` on a `.wav`/`.flac` | `SIH_DATA_ROOT` wrong or the copy is incomplete. Re-check the file count in step 6. |
| Training seems frozen | Check `nvidia-smi` for GPU use and `tail train_wide48.log`. A step line appears every ~20 s; validation between epochs is quiet for about a minute. |
| Machine rebooted | Run the `nohup` line from step 9 again. Nothing is lost beyond the epoch in progress. |
| It says `STOP_wide48 present` | That file exists; `rm STOP_wide48` and start again. |

**If something isn't covered here, send the last 30 lines of the log:**

```bash
tail -30 train_wide48.log
```

That is almost always enough to diagnose it remotely.
