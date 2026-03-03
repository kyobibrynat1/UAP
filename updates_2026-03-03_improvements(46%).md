# Updates (2026-03-03) - Fooling Rate Improvements

## Summary
- Adjusted default training parameters to increase fooling rate while keeping outputs visually similar.

## What Was Changed
- Increased training iterations to improve optimization without adding extra noise.
- Kept noise budget moderate and slightly raised alpha blending to strengthen perturbations with limited visual impact.
- Increased negative-prompt loss weight to push images away from semantic matches.
- Increased caption diversity per image to improve robustness.
- Set a moderate fooling drop threshold to avoid over-aggressive criteria.

## Files Changed
- [python/clip_uap_generator.py](python/clip_uap_generator.py)
