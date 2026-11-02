# Visual matching foundations

Issue [#5](https://github.com/JRealValdes/jarvis-media-processor/issues/5). Strategy context: [#16](https://github.com/JRealValdes/jarvis-media-processor/issues/16).

## Approach

1. Sample frames from **base** and **donor** with FFmpeg (`fps=JMP_VIDEO_SAMPLE_FPS`, default 1.0).
2. Normalize each frame: RGB, optional letterbox/pillarbox crop, downscale.
3. Compute perceptual hash (`imagehash.phash`).
4. For each base frame, keep donor candidates within a Hamming-distance threshold and convert distance → confidence.

## Cost / quality notes

| Aspect | Notes |
|--------|--------|
| Cost | Dominated by FFmpeg decode + PNG write. At 1 fps, a 2h film ≈ 7200 frames/side — fine for offline jobs, heavy for interactive use. Raise interval (lower fps) for coarse passes. |
| Strengths | Cheap, deterministic, decent under mild scale/letterbox/compression. |
| Limits | Channel logos / burned-in overlays / heavy color grading can inflate distances. Cropping bars helps AR differences; logos are **not** magically ignored yet (optional ROI later). |
| Next | Fuse with acoustic landmarks in #6; no heavy embeddings unless pHash proves insufficient on real pairs. |

## CLI

```bash
uv run jmp match-visual path/to/base.mkv path/to/donor.mkv -w .jmp/visual-match
```
