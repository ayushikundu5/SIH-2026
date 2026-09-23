# Results - wide32_mid

- generated: 2026-09-22T00:17:21+00:00
- git: `not-a-repo`
- test set: `C:\SIH26052_data\testset` (frozen, seed 20260827)
- device: `cpu`
- PESQ available: True
- clips: 720 per method
- eval workers: 4 (RTF column is under parallel load - see results/bench.json)

Targets: PESQ > 2.5 (scored on WIDEBAND PESQ; narrowband shown for reference), STOI > 0.85, SNR > 15.0 dB (scored both as OUTPUT SNR and as SNR GAIN), RTF < 0.5.

## Burst-local performance (impulsive categories)

SI-SDR gain measured INSIDE the gunfire/explosion bursts only. A whole-clip score is diluted by the ~88% of each clip containing no transient, so a model that removes none of the gunfire can still look acceptable overall. This column is the one that answers the question.

| category | method | burst SI-SDR gain | whole-clip SI-SDR gain | burst % of clip |
|---|---|---|---|---|
| artillery | `onnx:C:/SIH26052_data/mid/model_wide32_mid_simple.onnx` | **+8.30** | +8.60 | 20.3% |
| gunshot | `onnx:C:/SIH26052_data/mid/model_wide32_mid_simple.onnx` | **+8.17** | +7.96 | 18.7% |

### artillery

| method | PESQ-WB | PESQ-NB | STOI | SI-SDR gain | output SNR | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|---|---|
| `onnx:C:/SIH26052_data/mid/model_wide32_mid_simple.onnx` | 1.804 | 2.407 | 0.850 | +8.60 | +9.82 | +9.28 | 0.6379 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFFAIL |

### babble

| method | PESQ-WB | PESQ-NB | STOI | SI-SDR gain | output SNR | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|---|---|
| `onnx:C:/SIH26052_data/mid/model_wide32_mid_simple.onnx` | 1.943 | 2.450 | 0.846 | +5.18 | +10.69 | +6.02 | 0.6436 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFFAIL |

### engine

| method | PESQ-WB | PESQ-NB | STOI | SI-SDR gain | output SNR | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|---|---|
| `onnx:C:/SIH26052_data/mid/model_wide32_mid_simple.onnx` | 2.204 | 2.772 | 0.887 | +6.24 | +12.24 | +6.65 | 0.6581 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |

### gunshot

| method | PESQ-WB | PESQ-NB | STOI | SI-SDR gain | output SNR | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|---|---|
| `onnx:C:/SIH26052_data/mid/model_wide32_mid_simple.onnx` | 2.042 | 2.589 | 0.866 | +7.96 | +11.28 | +8.44 | 0.6419 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |

### rotor

| method | PESQ-WB | PESQ-NB | STOI | SI-SDR gain | output SNR | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|---|---|
| `onnx:C:/SIH26052_data/mid/model_wide32_mid_simple.onnx` | 2.135 | 2.720 | 0.887 | +5.49 | +12.02 | +5.91 | 0.6242 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |

### siren

| method | PESQ-WB | PESQ-NB | STOI | SI-SDR gain | output SNR | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|---|---|
| `onnx:C:/SIH26052_data/mid/model_wide32_mid_simple.onnx` | 2.201 | 2.786 | 0.903 | +5.77 | +12.72 | +6.03 | 0.6490 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |

## All targets by INPUT SNR (all categories pooled)

| method | input SNR | n | PESQ-WB | PESQ-NB | STOI | output SNR | SNR gain |
|---|---|---|---|---|---|---|---|
| `onnx:C:/SIH26052_data/mid/model_wide32_mid_simple.onnx` | <0 dB | 170 | 1.427 | 1.904 | 0.755 | +5.70 | +10.35 |
| `onnx:C:/SIH26052_data/mid/model_wide32_mid_simple.onnx` | 0-5 dB | 232 | 1.843 | 2.435 | 0.867 | +10.13 | +7.62 |
| `onnx:C:/SIH26052_data/mid/model_wide32_mid_simple.onnx` | 5-10 dB | 169 | 2.240 | 2.888 | 0.921 | +13.05 | +5.55 |
| `onnx:C:/SIH26052_data/mid/model_wide32_mid_simple.onnx` | 10-15 dB | 95 | 2.758 | 3.334 | 0.958 | +17.24 | +4.89 |
| `onnx:C:/SIH26052_data/mid/model_wide32_mid_simple.onnx` | >15 dB | 54 | 3.122 | 3.582 | 0.971 | +20.16 | +2.75 |

## SNR gain by INPUT SNR (dB)

The >15 dB target is only reachable in the low-input-SNR regime; at high input SNR there is little noise left to remove.

| method | <0 dB | 0-5 dB | 5-10 dB | 10-15 dB | >15 dB |
|---|---|---|---|---|---|
| `onnx:C:/SIH26052_data/mid/model_wide32_mid_simple.onnx` | +10.35 (n=170) | +7.62 (n=232) | +5.55 (n=169) | +4.89 (n=95) | +2.75 (n=54) |