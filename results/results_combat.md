# Results - combat

- generated: 2026-09-24T13:07:56+00:00
- git: `not-a-repo`
- test set: `C:\SIH26052_data\testset_combat` (frozen, seed 20260827)
- device: `cpu`
- PESQ available: True
- clips: 150 per method
- eval workers: 11 (RTF column is under parallel load - see results/bench.json)

Targets: PESQ > 2.5 (scored on WIDEBAND PESQ; narrowband shown for reference), STOI > 0.85, SNR > 15.0 dB (scored both as OUTPUT SNR and as SNR GAIN), RTF < 0.5.

### combat_real

| method | PESQ-WB | PESQ-NB | STOI | SI-SDR gain | output SNR | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|---|---|
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | 1.773 | 2.331 | 0.846 | +4.70 | +10.87 | +5.15 | 0.8816 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_combat32_simple.onnx` | 1.977 | 2.530 | 0.865 | +6.13 | +12.15 | +6.43 | 0.7035 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_fresh32_simple.onnx` | 2.002 | 2.551 | 0.869 | +6.25 | +12.30 | +6.59 | 0.7304 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_short24_simple.onnx` | 1.880 | 2.391 | 0.852 | +5.35 | +11.30 | +5.58 | 0.6887 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_simple.onnx` | 1.858 | 2.381 | 0.850 | +5.36 | +11.46 | +5.74 | 0.5580 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_wide32_simple.onnx` | 1.879 | 2.445 | 0.860 | +5.86 | +11.94 | +6.23 | 1.1355 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFFAIL |
| `unprocessed` | 1.298 | 1.718 | 0.783 | +0.00 | +5.72 | +0.00 | 0.0000 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |
| `wiener` | 1.339 | 1.769 | 0.782 | +0.53 | +6.42 | +0.70 | 0.0289 | PESQFAIL STOIFAIL SNRoutFAIL SNRgainFAIL RTFPASS |

## All targets by INPUT SNR (all categories pooled)

| method | input SNR | n | PESQ-WB | PESQ-NB | STOI | output SNR | SNR gain |
|---|---|---|---|---|---|---|---|
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | <0 dB | 23 | 1.321 | 1.775 | 0.727 | +5.19 | +8.80 |
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | 0-5 dB | 55 | 1.503 | 2.037 | 0.814 | +8.95 | +6.48 |
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | 5-10 dB | 34 | 1.865 | 2.463 | 0.873 | +12.29 | +4.84 |
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | 10-15 dB | 21 | 2.225 | 2.918 | 0.936 | +14.59 | +2.48 |
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | >15 dB | 17 | 2.520 | 3.045 | 0.941 | +17.29 | -0.18 |
| `onnx:artifacts/model_combat32_simple.onnx` | <0 dB | 23 | 1.431 | 1.908 | 0.758 | +6.63 | +10.24 |
| `onnx:artifacts/model_combat32_simple.onnx` | 0-5 dB | 55 | 1.641 | 2.208 | 0.835 | +9.77 | +7.30 |
| `onnx:artifacts/model_combat32_simple.onnx` | 5-10 dB | 34 | 2.066 | 2.671 | 0.890 | +13.34 | +5.89 |
| `onnx:artifacts/model_combat32_simple.onnx` | 10-15 dB | 21 | 2.490 | 3.083 | 0.946 | +16.04 | +3.93 |
| `onnx:artifacts/model_combat32_simple.onnx` | >15 dB | 17 | 2.990 | 3.446 | 0.959 | +20.11 | +2.64 |
| `onnx:artifacts/model_fresh32_simple.onnx` | <0 dB | 23 | 1.432 | 1.916 | 0.764 | +6.97 | +10.58 |
| `onnx:artifacts/model_fresh32_simple.onnx` | 0-5 dB | 55 | 1.654 | 2.220 | 0.840 | +9.82 | +7.36 |
| `onnx:artifacts/model_fresh32_simple.onnx` | 5-10 dB | 34 | 2.090 | 2.704 | 0.894 | +13.33 | +5.87 |
| `onnx:artifacts/model_fresh32_simple.onnx` | 10-15 dB | 21 | 2.547 | 3.130 | 0.947 | +16.17 | +4.06 |
| `onnx:artifacts/model_fresh32_simple.onnx` | >15 dB | 17 | 3.052 | 3.463 | 0.959 | +20.71 | +3.23 |
| `onnx:artifacts/model_short24_simple.onnx` | <0 dB | 23 | 1.338 | 1.751 | 0.731 | +6.01 | +9.62 |
| `onnx:artifacts/model_short24_simple.onnx` | 0-5 dB | 55 | 1.554 | 2.057 | 0.820 | +8.81 | +6.35 |
| `onnx:artifacts/model_short24_simple.onnx` | 5-10 dB | 34 | 1.925 | 2.506 | 0.878 | +12.40 | +4.94 |
| `onnx:artifacts/model_short24_simple.onnx` | 10-15 dB | 21 | 2.359 | 2.980 | 0.941 | +15.23 | +3.12 |
| `onnx:artifacts/model_short24_simple.onnx` | >15 dB | 17 | 2.987 | 3.381 | 0.955 | +19.41 | +1.93 |
| `onnx:artifacts/model_simple.onnx` | <0 dB | 23 | 1.371 | 1.788 | 0.731 | +6.17 | +9.78 |
| `onnx:artifacts/model_simple.onnx` | 0-5 dB | 55 | 1.549 | 2.062 | 0.818 | +9.11 | +6.65 |
| `onnx:artifacts/model_simple.onnx` | 5-10 dB | 34 | 1.918 | 2.522 | 0.876 | +12.60 | +5.14 |
| `onnx:artifacts/model_simple.onnx` | 10-15 dB | 21 | 2.337 | 2.954 | 0.940 | +15.17 | +3.06 |
| `onnx:artifacts/model_simple.onnx` | >15 dB | 17 | 2.803 | 3.229 | 0.952 | +19.34 | +1.87 |
| `onnx:artifacts/model_wide32_simple.onnx` | <0 dB | 23 | 1.371 | 1.831 | 0.746 | +6.67 | +10.27 |
| `onnx:artifacts/model_wide32_simple.onnx` | 0-5 dB | 55 | 1.561 | 2.135 | 0.830 | +9.46 | +7.00 |
| `onnx:artifacts/model_wide32_simple.onnx` | 5-10 dB | 34 | 1.956 | 2.591 | 0.884 | +13.18 | +5.73 |
| `onnx:artifacts/model_wide32_simple.onnx` | 10-15 dB | 21 | 2.374 | 2.990 | 0.944 | +15.95 | +3.84 |
| `onnx:artifacts/model_wide32_simple.onnx` | >15 dB | 17 | 2.830 | 3.314 | 0.958 | +19.68 | +2.20 |
| `unprocessed` | <0 dB | 23 | 1.092 | 1.362 | 0.628 | -3.61 | +0.00 |
| `unprocessed` | 0-5 dB | 55 | 1.134 | 1.488 | 0.732 | +2.46 | +0.00 |
| `unprocessed` | 5-10 dB | 34 | 1.253 | 1.728 | 0.817 | +7.46 | +0.00 |
| `unprocessed` | 10-15 dB | 21 | 1.558 | 2.183 | 0.907 | +12.11 | +0.00 |
| `unprocessed` | >15 dB | 17 | 1.881 | 2.349 | 0.934 | +17.48 | +0.00 |
| `wiener` | <0 dB | 23 | 1.107 | 1.384 | 0.626 | -2.86 | +0.75 |
| `wiener` | 0-5 dB | 55 | 1.146 | 1.508 | 0.733 | +3.24 | +0.78 |
| `wiener` | 5-10 dB | 34 | 1.285 | 1.789 | 0.817 | +8.03 | +0.58 |
| `wiener` | 10-15 dB | 21 | 1.633 | 2.276 | 0.907 | +12.82 | +0.71 |
| `wiener` | >15 dB | 17 | 2.022 | 2.466 | 0.932 | +18.15 | +0.67 |

## SNR gain by INPUT SNR (dB)

The >15 dB target is only reachable in the low-input-SNR regime; at high input SNR there is little noise left to remove.

| method | <0 dB | 0-5 dB | 5-10 dB | 10-15 dB | >15 dB |
|---|---|---|---|---|---|
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | +8.80 (n=23) | +6.48 (n=55) | +4.84 (n=34) | +2.48 (n=21) | -0.18 (n=17) |
| `onnx:artifacts/model_combat32_simple.onnx` | +10.24 (n=23) | +7.30 (n=55) | +5.89 (n=34) | +3.93 (n=21) | +2.64 (n=17) |
| `onnx:artifacts/model_fresh32_simple.onnx` | +10.58 (n=23) | +7.36 (n=55) | +5.87 (n=34) | +4.06 (n=21) | +3.23 (n=17) |
| `onnx:artifacts/model_short24_simple.onnx` | +9.62 (n=23) | +6.35 (n=55) | +4.94 (n=34) | +3.12 (n=21) | +1.93 (n=17) |
| `onnx:artifacts/model_simple.onnx` | +9.78 (n=23) | +6.65 (n=55) | +5.14 (n=34) | +3.06 (n=21) | +1.87 (n=17) |
| `onnx:artifacts/model_wide32_simple.onnx` | +10.27 (n=23) | +7.00 (n=55) | +5.73 (n=34) | +3.84 (n=21) | +2.20 (n=17) |
| `unprocessed` | +0.00 (n=23) | +0.00 (n=55) | +0.00 (n=34) | +0.00 (n=21) | +0.00 (n=17) |
| `wiener` | +0.75 (n=23) | +0.78 (n=55) | +0.58 (n=34) | +0.71 (n=21) | +0.67 (n=17) |