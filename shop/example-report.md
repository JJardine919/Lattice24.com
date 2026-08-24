# Sensor / process report

File `indpen_test2.csv` · 24,000 rows · 15 usable channels
Engine checksum 34.031437 (required 34.031437) · 2026-08-06

Read as 2400 windows of 8 consecutive samples across 15 channels: Aeration, SugarFeed, AcidFlow, BaseFlow, CoolWater, HeatWater, WaterInj, AirPressure, DumpedBroth, Substrate, DissolvedO2, Penicillin, Volume, Weight, pH

Grouped by `batch` into 40 runs. Readings are averaged within each run before runs are compared, because windows inside one run are not independent of each other.

## Encoder check

| bin | share |
|---|---|
| < -0.2% | 0.180 |
| -0.2..-0.05% | 0.053 |
| flat | 0.456 |
| +0.05..0.2% | 0.126 |
| > +0.2% | 0.186 |

End-bin mass **0.3653** on the standard scale.
 Healthy — the channel has room to move. (Saturation would be >0.9.)

## Result

Comparing **fault** vs **ok**, primary channel is entropy (named before reading).

Unit of comparison: **40 runs**, not individual windows.

| channel | difference | p | |
|---|---|---|---|
| entropy **primary** | -0.00399 | 0.0455 | separates |
| chaos secondary | -0.00664 | 0.0002 | separates |
| control secondary | +0.00155 | 0.7083 | — |
| anomaly secondary | +0.00576 | 0.0277 | separates |

p is a permutation test on the labels, 4000 draws. A secondary channel does not rescue a null primary — with four channels, roughly one looks significant by luck.

## Read this before acting on the above

- Only 15 channels supplied; the engine takes 16. The remaining slots are zero-filled, which weakens the read.
- **The two groups are blocked in time, not interleaved.** The label changes 1× across 40 runs; if the conditions were mixed through the campaign you would expect about 15. So every 'ok' sits on one side of the run order and every 'fault' on the other, and a difference above may be *when* the data was recorded rather than the condition itself. **No analysis can separate those after the fact** — it needs fault and ok runs interleaved in time. Treat the result as suggestive, not settled, and if you have healthy runs from the same period as the flagged ones, send those and we re-run.
- Worth checking on your side: is there anything else that changed between the early and late runs — a recipe revision, a control-strategy change, a maintenance event? Anything blocked the same way is indistinguishable from the condition in this data.

## What this does not tell you

- It does not say a channel is *correct*, only whether it moves.
- It does not replace your instruments or your calibration schedule.
- Nothing here was fitted to your data, so nothing is tuned to flatter it. The same pipeline runs on every file unchanged.

---
Lattice24 · James Jardine · ORCID 0009-0004-9073-7192