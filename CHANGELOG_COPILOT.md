# Changes made by GitHub Copilot

## Overview

This file summarizes the changes applied during this session.

## python/clip_integration.py

- Added CUDA performance settings and optional mixed precision (`use_fp16`) to speed up CLIP inference on GPU.
- Enabled autocast during image/text encoding when running on CUDA.
- Ensured input tensors are converted to fp16 when mixed precision is enabled.

## python/clip_uap_generator.py

- Switched optimization to batched gradients instead of per-image updates.
- Added momentum-based updates and configurable steps per iteration.
- Added batched text embedding and image stacking for faster training.
- Reworked evaluation to compare against per-image baseline similarity and compute fooling rate by similarity drop ratio.
- Added adaptive learning rate decay on plateau with early stopping.
- Added new tuning parameters: `steps_per_iteration`, `momentum`, `fooling_drop_ratio`, `lr_decay`, `min_learning_rate`, `plateau_patience`, `min_fooling_improve`.
