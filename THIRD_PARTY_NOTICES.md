# Third-party notices

## GTCRN

`src/models/gtcrn.py` and `third_party/gtcrn/` are taken from
<https://github.com/Xiaobin-Rong/gtcrn>, MIT licence, copyright (c) Xiaobin Rong.
The full licence text is preserved at `src/models/LICENSE.gtcrn`.

Rong et al., *"GTCRN: A Speech Enhancement Model Requiring Ultralow
Computational Resources"*, ICASSP 2024.

`checkpoints/pretrained/` contains the author's released weights
(`model_trained_on_dns3.tar`, `model_trained_on_vctk.tar`), used as the
initialisation for fine-tuning and as evaluation baselines.

## SEtrain

The training recipe in `configs/train_wide.yaml`, `train_fresh32.yaml`,
`train_short24.yaml` and `train_short32.yaml` — lr 1e-3, warmup then cosine to
1e-6, grad clip 3.0, loss weights 70/30/1 — follows the upstream GTCRN training
template at <https://github.com/Xiaobin-Rong/SEtrain>, MIT licence, copyright
(c) 2025 Rong Xiaobin.

**No SEtrain code is used or vendored in this repository.** A local reference
checkout may exist at `third_party/SEtrain/` on a development machine; it is
gitignored, because nothing here imports it and it carries a 2.7 MB DNSMOS model
we do not use. The dependency is on the published hyperparameters, not the code,
and this entry exists so that borrowing is stated rather than implied.

## Datasets

Not redistributed here — downloaded by `scripts/download_*.sh`.

| Dataset | Licence |
|---|---|
| LibriSpeech (OpenSLR 12) | CC BY 4.0 |
| MUSAN (OpenSLR 17) | CC BY 4.0 |
| Room Impulse Responses (OpenSLR 28) | Apache 2.0 |
| Gunshot/Gunfire Audio Dataset (Zenodo 7004819) | CC BY 4.0 |
| UrbanSound8K (Zenodo 1203745) | CC BY-NC 4.0 |
| ESC-50 | CC BY-NC 3.0 |
| VoiceBank-DEMAND (Edinburgh DataShare 10283/2791) | see record EULA |

**ESC-50 and UrbanSound8K are non-commercial.** Suitable for research and
competition use; they must be replaced with CC-BY sources before any commercial
or procurement use.

## Metrics

- `pesq` — ITU-T P.862 reference implementation
- `pystoi` — reference STOI/ESTOI implementation
