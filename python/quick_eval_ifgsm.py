"""
Quick I-FGSM Evaluation Script
Small-scale training run to validate algorithm correctness before full 5K training

Purpose:
    - Uses ~50 images instead of 5,000 to finish in minutes
    - Validates gradient flow, loss convergence, and similarity drop
    - Confirms I-FGSM update rule is working correctly
    - Prints detailed per-iteration diagnostics

Usage:
    python quick_eval_ifgsm.py
    python quick_eval_ifgsm.py --images 100 --iterations 5
"""

import numpy as np
import torch
import clip
import os
import sys
import argparse
import time
from typing import Dict, List, Optional
from tqdm import tqdm
from datetime import datetime

from clip_integration import CLIPModelWrapper
from coco_loader import COCODataLoader


class IFGSMEvaluator:
    """
    Small-scale I-FGSM evaluator for algorithm validation.
    
    Runs a lightweight training loop and checks:
      1. Gradients are non-zero and flowing
      2. Loss decreases (or similarity drops) across iterations
      3. Perturbation norm grows and stays within bounds
      4. Fooling rate improves over iterations
    """

    def __init__(self, clip_model: CLIPModelWrapper, data_loader: COCODataLoader):
        self.clip_model = clip_model
        self.data_loader = data_loader
        self.device = clip_model.device
        self.model = clip_model.get_model()
        self.preprocess = clip_model.get_preprocess()

    # ------------------------------------------------------------------
    # Core helpers (mirror clip_uap_generator.py logic)
    # ------------------------------------------------------------------
    def _semantic_loss(self, image_tensor: torch.Tensor,
                       text_features: torch.Tensor) -> torch.Tensor:
        """cos(f_img(I+v), f_txt(T))"""
        image_features = self.model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (image_features * text_features).sum(dim=1)
        return similarity.mean()

    def _apply_alpha_blending(self, original: torch.Tensor,
                               perturbation: torch.Tensor,
                               alpha: float) -> torch.Tensor:
        """I' = (1 - α)·I + α·(I + v)"""
        perturbed = torch.clamp(original + perturbation, 0, 1)
        blended = (1 - alpha) * original + alpha * perturbed
        return torch.clamp(blended, 0, 1)

    def _project(self, v: torch.Tensor, xi: float,
                 norm_type: str = "inf") -> torch.Tensor:
        if norm_type == "inf":
            return torch.clamp(v, -xi, xi)
        else:
            norm = torch.norm(v)
            return v * (xi / norm) if norm > xi else v

    # ------------------------------------------------------------------
    # Evaluation run
    # ------------------------------------------------------------------
    def run(self,
            num_images: int = 50,
            num_iterations: int = 5,
            xi: float = 32 / 255,
            learning_rate: float = 8 / 255,
            alpha: float = 1.0,
            use_annotations: bool = True,
            seed: int = 42) -> Dict:
        """
        Execute small-scale I-FGSM and return diagnostics.

        Args:
            num_images:     Number of images to train on (default 50)
            num_iterations: Optimisation passes (default 5)
            xi:             L∞ perturbation bound
            learning_rate:  Step size α
            alpha:          Alpha-blending factor
            use_annotations: Use COCO captions
            seed:           Reproducibility seed

        Returns:
            Dictionary with per-iteration diagnostics and pass/fail verdict.
        """
        header = "=" * 62
        print(header)
        print("  QUICK I-FGSM EVALUATION")
        print(f"  Images: {num_images}  |  Iterations: {num_iterations}")
        print(f"  xi={xi:.4f}  lr={learning_rate:.4f}  alpha={alpha}")
        print(header)

        # ----- 1. Load a small batch -----
        image_paths = self.data_loader.get_mini_batch(num_images, seed=seed)
        text_descriptions = self.data_loader.create_text_descriptions_for_batch(
            num_images, use_annotations=use_annotations, image_paths=image_paths
        )

        print(f"\n[1/4] Loading {num_images} images ...")
        dataset: List[torch.Tensor] = []
        valid_texts: List[str] = []
        for img_path, desc in zip(image_paths, text_descriptions):
            try:
                img_tensor = self.clip_model._load_image(img_path)
                dataset.append(img_tensor)
                valid_texts.append(desc)
            except Exception:
                continue
        actual_n = len(dataset)
        print(f"     Loaded {actual_n} images successfully")

        if actual_n == 0:
            print("ERROR: No images loaded. Check data/MS-COCO/val2017 path.")
            return {"pass": False, "reason": "no images"}

        # ----- 2. Pre-compute text features -----
        print("[2/4] Encoding text features ...")
        text_features_list = []
        for desc in valid_texts:
            emb = self.clip_model.get_text_embeddings(desc)
            text_features_list.append(torch.from_numpy(emb).to(self.device))
        text_features = torch.stack(text_features_list)

        # ----- 3. Compute baseline similarities -----
        print("[3/4] Computing baseline similarities ...")
        baseline_sims = []
        with torch.no_grad():
            for idx in range(actual_n):
                img_feat = self.model.encode_image(dataset[idx])
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                sim = (100.0 * (img_feat * text_features[idx:idx+1]).sum()).item()
                baseline_sims.append(sim)
        baseline_mean = np.mean(baseline_sims)
        print(f"     Baseline mean similarity: {baseline_mean:.2f}%")

        # ----- 4. I-FGSM optimisation loop -----
        print(f"[4/4] Running I-FGSM optimisation ({num_iterations} iterations) ...\n")

        v = torch.zeros(1, 3, 224, 224, device=self.device)
        history = []

        for it in range(1, num_iterations + 1):
            t0 = time.time()
            grad_norms = []
            losses = []

            for idx in range(actual_n):
                img = dataset[idx]
                text_feat = text_features[idx:idx + 1]

                v_grad = v.clone().detach().requires_grad_(True)
                blended = self._apply_alpha_blending(img, v_grad, alpha)
                loss = self._semantic_loss(blended, text_feat)
                loss.backward()

                with torch.no_grad():
                    grad = v_grad.grad
                    grad_norms.append(grad.norm().item())
                    losses.append(loss.item())
                    # Gradient DESCENT: subtract to minimize cosine similarity
                    v = v - learning_rate * grad.sign()
                    v = self._project(v, xi, "inf")

            elapsed = time.time() - t0

            # Evaluate post-iteration
            post_sims = []
            fooled = 0
            with torch.no_grad():
                for idx in range(actual_n):
                    blended = self._apply_alpha_blending(dataset[idx], v, alpha)
                    img_feat = self.model.encode_image(blended)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                    sim = (100.0 * (img_feat * text_features[idx:idx+1]).sum()).item()
                    post_sims.append(sim)
                    if sim < baseline_sims[idx] * 0.7:
                        fooled += 1

            post_mean = np.mean(post_sims)
            fooling_rate = fooled / actual_n
            v_linf = torch.max(torch.abs(v)).item()
            v_l2 = torch.norm(v).item()
            mean_grad = np.mean(grad_norms)
            mean_loss = np.mean(losses)
            sim_drop = baseline_mean - post_mean

            record = {
                "iteration": it,
                "mean_loss": mean_loss,
                "mean_grad_norm": mean_grad,
                "v_linf": v_linf,
                "v_l2": v_l2,
                "baseline_sim": baseline_mean,
                "protected_sim": post_mean,
                "sim_drop": sim_drop,
                "fooling_rate": fooling_rate,
                "elapsed_s": elapsed,
            }
            history.append(record)

            # Pretty-print
            print(f"  Iter {it}/{num_iterations}  "
                  f"loss={mean_loss:.4f}  "
                  f"grad={mean_grad:.6f}  "
                  f"‖v‖∞={v_linf:.4f}  "
                  f"sim={post_mean:.1f}%  "
                  f"drop={sim_drop:+.1f}%  "
                  f"fool={fooling_rate*100:.1f}%  "
                  f"({elapsed:.1f}s)")

        # ----- Summary & Verdict -----
        print(f"\n{header}")
        print("  EVALUATION SUMMARY")
        print(header)

        final = history[-1]
        first = history[0]

        checks = {
            "gradients_nonzero": first["mean_grad_norm"] > 1e-8,
            "loss_decreased": final["mean_loss"] < first["mean_loss"] + 0.01,
            "similarity_dropped": final["sim_drop"] > 0.5,
            "perturbation_bounded": final["v_linf"] <= xi + 1e-6,
            "fooling_improved": final["fooling_rate"] >= first["fooling_rate"],
        }

        all_pass = True
        for check_name, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"  [{status}] {check_name}")

        print(f"\n  Baseline similarity : {baseline_mean:.2f}%")
        print(f"  Final similarity    : {final['protected_sim']:.2f}%")
        print(f"  Total sim drop      : {final['sim_drop']:+.2f}%")
        print(f"  Final fooling rate  : {final['fooling_rate']*100:.1f}%")
        print(f"  Perturbation ‖v‖∞   : {final['v_linf']:.6f}  (bound={xi:.6f})")
        print(f"  Perturbation ‖v‖₂   : {final['v_l2']:.4f}")

        if all_pass:
            print(f"\n  ✓ ALL CHECKS PASSED — I-FGSM is working correctly.")
            print(f"    Safe to proceed with full-scale training (5K images, 10 iterations).")
        else:
            print(f"\n  ✗ SOME CHECKS FAILED — review diagnostics above.")
            failed = [k for k, v in checks.items() if not v]
            for f in failed:
                if f == "gradients_nonzero":
                    print(f"    → Gradients are zero. CLIP model may not be producing gradients.")
                    print(f"      Check that model.encode_image is differentiable w.r.t. perturbation.")
                elif f == "similarity_dropped":
                    print(f"    → Similarity did not drop meaningfully.")
                    print(f"      Try increasing xi or learning_rate, or check annotation quality.")
                elif f == "loss_decreased":
                    print(f"    → Loss did not decrease. Optimisation may be diverging.")
                    print(f"      Try reducing learning_rate.")
                elif f == "fooling_improved":
                    print(f"    → Fooling rate did not improve. May need more iterations or larger xi.")

        print(header)

        # Save small-scale results
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "data", "results")
        os.makedirs(results_dir, exist_ok=True)
        eval_path = os.path.join(results_dir, "quick_eval_history.npy")
        np.save(eval_path, history)
        print(f"\n  Saved evaluation history → {os.path.normpath(eval_path)}")

        return {"pass": all_pass, "checks": checks, "history": history}


def main():
    parser = argparse.ArgumentParser(
        description="Quick I-FGSM evaluation (small-scale)")
    parser.add_argument("--images", type=int, default=50,
                        help="Number of images (default: 50)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of iterations (default: 5)")
    parser.add_argument("--xi", type=float, default=32/255,
                        help="Perturbation bound (default: 32/255)")
    parser.add_argument("--lr", type=float, default=8/255,
                        help="Learning rate (default: 8/255)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Alpha blending (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    # Resolve annotation path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ann_path = os.path.normpath(
        os.path.join(script_dir, "..", "data", "MS-COCO",
                     "annotations", "captions_val2017.json"))

    print("Initializing ...")
    clip_model = CLIPModelWrapper(model_name="ViT-B/32")
    data_loader = COCODataLoader(annotations_file=ann_path)
    evaluator = IFGSMEvaluator(clip_model, data_loader)

    evaluator.run(
        num_images=args.images,
        num_iterations=args.iterations,
        xi=args.xi,
        learning_rate=args.lr,
        alpha=args.alpha,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
