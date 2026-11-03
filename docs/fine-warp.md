# Post-reconstruct fine-warp

After `jmp align` + `jmp reconstruct`, donor audio sits on the base timeline but can still show residual A/V lag (tenths of a second up to ~1 s in hard cross-encode / cross-language cases). `jmp fine-warp` estimates a slowly varying lag field `L(t)` against **base audio stream 0** and remaps the reconstructed WAV with `out[t] = recon[t + L(t)]` before mux.

This pass does **not** re-detect surplus ads or missing-base gaps. Structure comes from alignment; fine-warp only trims residual timing.

## Pipeline position

```text
align → reconstruct → fine-warp → mux
```

```bash
uv run jmp fine-warp path/to/base.mkv .jmp/reconstruct/donor_on_base.wav -w .jmp/fine-warp
uv run jmp mux path/to/base.mkv .jmp/fine-warp/donor_on_base_fine.wav -o out.mkv
```

Artifacts under the workspace:

| File | Role |
|------|------|
| `base_track0_fine_warp.wav` | Base track 0 extract (reference) |
| `donor_on_base_fine.wav` | Warped reconstructed audio |
| `fine_warp_plan.json` | Dense lag knots + notes (debug) |

## How it works (mental model)

1. **Comparable envelopes** — loudness-normalize both tracks, band-pass (~300–6000 Hz), then onset/flux energy. Shared FX and attacks correlate better than cross-language dialogue RMS.
2. **Dense local lag search** — every few seconds, find best lag within ±`JMP_FINE_WARP_MAX_LAG_SECONDS` (default 1 s) via envelope correlation; score margin vs lag=0 and a wider context window.
3. **Pillars** — keep only steps with enough confidence **and** absolute correlation (reject thin “best of a flat neighborhood” peaks).
4. **Interpolate** — low-confidence steps inherit adjacent pillars. They must **not** snap to a film-wide prior (a global prior can push the wrong direction).
5. **Apply** — continuous `L(t)` on the reconstructed WAV; mux as usual.

Design lessons (general, not title-specific): do not snap dialogue landmarks 1:1; do not leave accept/reject islands that fall back to `L=0` next to corrected regions; do not fall back to a single global lag for the whole runtime.

## Knobs (`JMP_FINE_WARP_*`)

See `Settings` in `jarvis_media_processor.config.settings`. Important ones:

| Variable | Default | Role |
|----------|---------|------|
| `MAX_LAG_SECONDS` | `1.0` | Search radius per window |
| `WINDOW_SECONDS` | `4.0` | Local correlation window |
| `STEP_SECONDS` | `2.0` | Dense grid step |
| `CONTEXT_WINDOW_SECONDS` | `10.0` | Big-picture improvement vs lag=0 |
| `CONFIDENCE_TAU` | `0.08` | Min confidence to be a pillar |
| `MIN_PILLAR_SCORE` | `0.4` | Min absolute correlation to be a pillar |
| `SMOOTH_SECONDS` | `20.0` | Smooth among pillars only |
| `BAND_LOW_HZ` / `BAND_HIGH_HZ` | `300` / `6000` | Band before onset envelope |

## When another title goes wrong

Inspect `fine_warp_plan.json` and compare base track 0 vs `donor_on_base.wav` (pre-warp) and `donor_on_base_fine.wav` (post-warp) around the bad timestamp.

| Symptom | Likely cause | Where to look / what to try |
|---------|--------------|-----------------------------|
| Alternating “perfect / early / late” | Sparse islands or wrong fallback (prior / zero gaps) | Knot density; ensure weak regions **interpolate pillars**, not a film-wide prior |
| Whole stretch late or early by ~constant | Under/over pillar lag; prior pull too strong | Raw local lag at a clear FX hit vs applied `lag_seconds`; lower `PRIOR_WEIGHT` |
| Still off by ~1 s after fine-warp | Residual larger than search, or **align/reconstruct** placed the chunk wrong | Wide lag sweep on **pre-warp** WAV; if true lag ≫ max or correlation is flat, fix alignment first |
| Dialogue scenes worse, FX scenes fine | Envelope still speech-dominated | Raise band low / rely more on onsets; raise `MIN_PILLAR_SCORE` |
| Almost no change vs reconstruct | Too few pillars | Plan notes (`Lag pillars=N/...`); loosen tau/score slightly or check extract SR/channels |
| Correction in the **wrong direction** | Sign of `L` misapplied or fallback to a prior with opposite sign | Confirm `out[t]=recon[t+L]` with a known late/early click; never gate weak steps to a global median of mixed signs |
| Good lock just before a line, line itself late | Smooth/prior bled away from a strong pillar into a weak window | Pillar interp should hold ~pillar lag across the weak span; check knots at ±20 s |

### Debug recipe

1. Note the timecode (base timeline).
2. On **reconstruct** (not fine) audio, sweep local lag ±1 s with the same onset envelopes; record best lag and score vs lag=0.
3. Read applied `L` from `fine_warp_plan.json` at that time.
4. If (2) is strong and (3) disagrees → field / pillar / prior bug. If (2) is weak/flat → do not trust fine-warp there; improve align or accept content mismatch.
5. Re-mux only after `donor_on_base_fine.wav` changes.

## Related

- Product / sync strategy: GitHub [#16](https://github.com/JRealValdes/jarvis-media-processor/issues/16)
- Fine-warp feature tracking: GitHub [#49](https://github.com/JRealValdes/jarvis-media-processor/issues/49)
