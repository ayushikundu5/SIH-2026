# Results - vbd_onnx

- generated: 2026-09-22T19:07:38+00:00
- git: `not-a-repo`
- test set: `C:\SIH26052_data\voicebank_demand` (frozen, seed None)
- device: `cpu`
- PESQ available: True
- clips: 824 per method
- eval workers: 11 (RTF column is under parallel load - see results/bench.json)

Targets: PESQ > 2.5 (scored on WIDEBAND PESQ; narrowband shown for reference), STOI > 0.85, SNR > 15.0 dB (scored both as OUTPUT SNR and as SNR GAIN), RTF < 0.5.

### voicebank_demand

| method | PESQ-WB | PESQ-NB | STOI | SI-SDR gain | output SNR | SNR gain | RTF | meets targets |
|---|---|---|---|---|---|---|---|---|
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | 2.511 | 3.214 | 0.915 | +7.32 | +15.87 | +7.42 | 0.5059 | PESQPASS STOIPASS SNRoutPASS SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_combat32_simple.onnx` | 2.334 | 3.180 | 0.911 | +9.10 | +17.52 | +9.08 | 0.6967 | PESQFAIL STOIPASS SNRoutPASS SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_lowsnr_simple.onnx` | 2.420 | 3.218 | 0.919 | +8.30 | +16.82 | +8.38 | 0.5036 | PESQFAIL STOIPASS SNRoutPASS SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_simple.onnx` | 2.387 | 3.252 | 0.921 | +7.83 | +16.34 | +7.90 | 0.5364 | PESQFAIL STOIPASS SNRoutPASS SNRgainFAIL RTFFAIL |
| `onnx:artifacts/model_wide32_simple.onnx` | 2.111 | 3.077 | 0.903 | +7.39 | +15.91 | +7.46 | 0.7144 | PESQFAIL STOIPASS SNRoutPASS SNRgainFAIL RTFFAIL |
| `unprocessed` | 1.968 | 2.879 | 0.921 | +0.00 | +8.45 | +0.00 | 0.0000 | PESQFAIL STOIPASS SNRoutFAIL SNRgainFAIL RTFPASS |

## All targets by INPUT SNR (all categories pooled)

| method | input SNR | n | PESQ-WB | PESQ-NB | STOI | output SNR | SNR gain |
|---|---|---|---|---|---|---|---|
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | <0 dB | 16 | 1.863 | 2.764 | 0.777 | +12.79 | +13.41 |
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | 0-5 dB | 202 | 2.083 | 2.753 | 0.880 | +11.77 | +10.48 |
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | 5-10 dB | 209 | 2.382 | 3.101 | 0.910 | +15.04 | +8.68 |
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | 10-15 dB | 206 | 2.682 | 3.421 | 0.933 | +17.40 | +6.18 |
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | >15 dB | 191 | 2.973 | 3.639 | 0.948 | +19.71 | +3.64 |
| `onnx:artifacts/model_combat32_simple.onnx` | <0 dB | 16 | 1.974 | 2.985 | 0.798 | +15.97 | +16.59 |
| `onnx:artifacts/model_combat32_simple.onnx` | 0-5 dB | 202 | 2.086 | 2.841 | 0.888 | +14.84 | +13.55 |
| `onnx:artifacts/model_combat32_simple.onnx` | 5-10 dB | 209 | 2.269 | 3.118 | 0.911 | +17.23 | +10.87 |
| `onnx:artifacts/model_combat32_simple.onnx` | 10-15 dB | 206 | 2.459 | 3.353 | 0.922 | +18.61 | +7.38 |
| `onnx:artifacts/model_combat32_simple.onnx` | >15 dB | 191 | 2.563 | 3.434 | 0.933 | +19.65 | +3.59 |
| `onnx:artifacts/model_lowsnr_simple.onnx` | <0 dB | 16 | 1.858 | 2.822 | 0.792 | +15.07 | +15.69 |
| `onnx:artifacts/model_lowsnr_simple.onnx` | 0-5 dB | 202 | 2.023 | 2.776 | 0.885 | +13.83 | +12.55 |
| `onnx:artifacts/model_lowsnr_simple.onnx` | 5-10 dB | 209 | 2.327 | 3.140 | 0.916 | +16.29 | +9.93 |
| `onnx:artifacts/model_lowsnr_simple.onnx` | 10-15 dB | 206 | 2.570 | 3.411 | 0.936 | +18.00 | +6.77 |
| `onnx:artifacts/model_lowsnr_simple.onnx` | >15 dB | 191 | 2.826 | 3.595 | 0.950 | +19.45 | +3.38 |
| `onnx:artifacts/model_simple.onnx` | <0 dB | 16 | 1.876 | 2.818 | 0.787 | +14.60 | +15.22 |
| `onnx:artifacts/model_simple.onnx` | 0-5 dB | 202 | 2.027 | 2.783 | 0.887 | +13.49 | +12.21 |
| `onnx:artifacts/model_simple.onnx` | 5-10 dB | 209 | 2.307 | 3.164 | 0.918 | +15.68 | +9.33 |
| `onnx:artifacts/model_simple.onnx` | 10-15 dB | 206 | 2.506 | 3.448 | 0.938 | +17.25 | +6.03 |
| `onnx:artifacts/model_simple.onnx` | >15 dB | 191 | 2.772 | 3.669 | 0.955 | +19.25 | +3.18 |
| `onnx:artifacts/model_wide32_simple.onnx` | <0 dB | 16 | 1.903 | 2.913 | 0.793 | +14.15 | +14.77 |
| `onnx:artifacts/model_wide32_simple.onnx` | 0-5 dB | 202 | 1.942 | 2.778 | 0.882 | +13.03 | +11.74 |
| `onnx:artifacts/model_wide32_simple.onnx` | 5-10 dB | 209 | 2.107 | 3.032 | 0.903 | +15.40 | +9.05 |
| `onnx:artifacts/model_wide32_simple.onnx` | 10-15 dB | 206 | 2.132 | 3.187 | 0.912 | +16.96 | +5.74 |
| `onnx:artifacts/model_wide32_simple.onnx` | >15 dB | 191 | 2.288 | 3.338 | 0.925 | +18.53 | +2.46 |
| `unprocessed` | <0 dB | 16 | 1.274 | 2.401 | 0.780 | -0.62 | +0.00 |
| `unprocessed` | 0-5 dB | 202 | 1.469 | 2.325 | 0.876 | +1.29 | +0.00 |
| `unprocessed` | 5-10 dB | 209 | 1.739 | 2.703 | 0.917 | +6.35 | +0.00 |
| `unprocessed` | 10-15 dB | 206 | 2.150 | 3.111 | 0.942 | +11.23 | +0.00 |
| `unprocessed` | >15 dB | 191 | 2.610 | 3.447 | 0.962 | +16.07 | +0.00 |

## SNR gain by INPUT SNR (dB)

The >15 dB target is only reachable in the low-input-SNR regime; at high input SNR there is little noise left to remove.

| method | <0 dB | 0-5 dB | 5-10 dB | 10-15 dB | >15 dB |
|---|---|---|---|---|---|
| `onnx:artifacts/gtcrn_dns3_simple.onnx` | +13.41 (n=16) | +10.48 (n=202) | +8.68 (n=209) | +6.18 (n=206) | +3.64 (n=191) |
| `onnx:artifacts/model_combat32_simple.onnx` | +16.59 (n=16) | +13.55 (n=202) | +10.87 (n=209) | +7.38 (n=206) | +3.59 (n=191) |
| `onnx:artifacts/model_lowsnr_simple.onnx` | +15.69 (n=16) | +12.55 (n=202) | +9.93 (n=209) | +6.77 (n=206) | +3.38 (n=191) |
| `onnx:artifacts/model_simple.onnx` | +15.22 (n=16) | +12.21 (n=202) | +9.33 (n=209) | +6.03 (n=206) | +3.18 (n=191) |
| `onnx:artifacts/model_wide32_simple.onnx` | +14.77 (n=16) | +11.74 (n=202) | +9.05 (n=209) | +5.74 (n=206) | +2.46 (n=191) |
| `unprocessed` | +0.00 (n=16) | +0.00 (n=202) | +0.00 (n=209) | +0.00 (n=206) | +0.00 (n=191) |