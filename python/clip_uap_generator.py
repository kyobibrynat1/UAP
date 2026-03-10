"""
CLIP Universal Adversarial Perturbation Generator
Core algorithm for generating semantic UAPs against CLIP ViT-B/32

Novel Contribution:
    This implementation adapts Iterative Fast Gradient Sign Method (I-FGSM)
    for Universal Adversarial Perturbations (UAP) targeting CLIP's multimodal
    latent space. The core optimization formula is:
    
    v_{t+1} = Π_ε { v_t - α · sign(∇_v cos(f_img(I + v_t), f_txt(T))) }
    
    where:
    - v_t: Universal perturbation at iteration t
    - α: Learning rate (step size)
    - Π_ε: Projection onto L_p ball of radius ε (L∞ or L2)
    - cos(·,·): Cosine similarity in CLIP's shared embedding space
    - f_img, f_txt: CLIP's image and text encoders
    
    Note: The subtraction (−) is critical — gradient DESCENT minimizes
    cosine similarity, which disrupts CLIP's semantic understanding.
    
Academic Foundations:
    - I-FGSM: Kurakin et al. (2016) - Basic Iterative Method
    - UAP: Moosavi-Dezfooli et al. (2017) - Universal Adversarial Perturbations
    - CLIP: Radford et al. (2021) - Vision-Language Model
    - COCO: Lin et al. (2014) - Training Dataset
    
Key Features:
1. Minimizing cosine similarity between images and text descriptions
2. Using alpha blending for mobile app compatibility
3. Memory-efficient mini-batch processing
"""

import numpy as np
import torch
import clip
import os
from tqdm import tqdm
from typing import Optional, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

from clip_integration import CLIPModelWrapper
from coco_loader import COCODataLoader


class UniversalPerturbationGenerator:
    """
    Generates Universal Adversarial Perturbations for CLIP model
    using semantic loss optimization with alpha blending
    """
    
    def __init__(self, 
                 clip_model: CLIPModelWrapper,
                 data_loader: COCODataLoader,
                 device: str = None):
        """
        Initialize UAP generator
        
        Args:
            clip_model: Initialized CLIP model wrapper
            data_loader: COCO data loader
            device: Torch device
        """
        self.clip_model = clip_model
        self.data_loader = data_loader
        self.device = device if device else clip_model.device
        
        # Get model for gradient computation
        self.model = clip_model.get_model()
        self.preprocess = clip_model.get_preprocess()
        
        print("[UAP] Generator initialized")
    
    def _semantic_loss(self, image_tensor: torch.Tensor, 
                       text_features: torch.Tensor) -> torch.Tensor:
        """
        Compute semantic loss: cosine similarity in CLIP's latent space
        
        This computes: cos(f_img(I + v_t), f_txt(T))
        where f_img and f_txt are CLIP's image and text encoders.
        
        Goal: MINIMIZE this loss to "cloak" the image from AI understanding
        (Targeting the vision-language alignment of Radford et al., 2021)
        Lower similarity = better protection
        
        Args:
            image_tensor: Batch of images (B, 3, 224, 224)
            text_features: Normalized text embeddings (B, 512)
            
        Returns:
            Mean cosine similarity (scalar tensor)
        """
        # Encode images
        image_features = self.model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute cosine similarity
        # Shape: (B, B) but we only need diagonal (paired similarities)
        similarity = (image_features * text_features).sum(dim=1)
        
        # Return mean similarity as loss
        return similarity.mean()
    
    def _apply_alpha_blending(self, 
                             original: torch.Tensor, 
                             perturbation: torch.Tensor,
                             alpha: float) -> torch.Tensor:
        """
        Apply alpha blending: Image_new = (1 - α) · Image + α · (Image + UAP)
        
        This ensures UAP is optimized for the EXACT blending used in mobile app
        
        Args:
            original: Original image tensor
            perturbation: UAP tensor
            alpha: Blending factor (0 to 1)
            
        Returns:
            Blended image tensor
        """
        # Add perturbation to original
        perturbed = torch.clamp(original + perturbation, 0, 1)
        
        # Alpha blend
        blended = (1 - alpha) * original + alpha * perturbed
        blended = torch.clamp(blended, 0, 1)
        
        return blended
    
    def _project_perturbation(self, 
                             perturbation: torch.Tensor,
                             xi: float,
                             norm_type: str = "inf") -> torch.Tensor:
        """
        Project perturbation onto L_p ball
        Ensures perturbation stays within bounded magnitude
        
        Args:
            perturbation: Current perturbation
            xi: Ball radius (max magnitude)
            norm_type: "inf" or "2"
            
        Returns:
            Projected perturbation
        """
        if norm_type == "inf":
            # L-infinity: element-wise clipping
            return torch.clamp(perturbation, -xi, xi)
        elif norm_type == "2":
            # L2: scale if norm exceeds xi
            norm = torch.norm(perturbation)
            if norm > xi:
                return perturbation * (xi / norm)
            return perturbation
        else:
            raise ValueError(f"Unsupported norm type: {norm_type}")
    
    def generate(self,
                 num_iterations: int = 10,
                 xi: float = 64/255,
                 learning_rate: float = 20/255,
                 alpha: float = 1.0,
                 delta: float = 0.2,
                 batch_size: int = 16,
                 use_momentum: bool = True,
                 momentum_decay: float = 1.0,
                 norm_type: str = "inf",
                 save_checkpoints: bool = True,
                 checkpoint_dir: str = "data/results",
                 use_annotations: bool = True,
                 seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate Universal Adversarial Perturbation
        
        NOVEL ALGORITHM - I-FGSM for CLIP Semantic Space
        
        Implements: v_{t+1} = Π_ε { v_t - α · sign(∇_v cos(f_img(I+v_t), f_txt(T))) }
        
        This is the core academic contribution: adapting I-FGSM (Kurakin et al., 2016)
        for universal perturbations (Moosavi-Dezfooli et al., 2017) targeting CLIP's
        vision-language alignment (Radford et al., 2021).
        
        Training uses the FULL dataset (all 5,000 val2017 images) every iteration
        to ensure the perturbation generalises across the entire distribution.
        Images and text features are loaded once and reused across iterations.
        
        Args:
            num_iterations: Number of full passes over the dataset
            xi: Perturbation magnitude bound (32/255 ≈ 0.125 recommended for measurable effect)
            learning_rate: Gradient descent step size (8/255 ≈ 0.031 for faster convergence)
            alpha: Alpha blending for mobile app (0.7 = 70% blend)
            delta: Target fooling rate (0.2 = 80% fooling)
            batch_size: Images per optimization step (higher = smoother gradients)
            use_momentum: Use MI-FGSM-style momentum for stronger UAPs
            momentum_decay: Momentum decay factor (1.0 = strong accumulation)
            norm_type: "inf" (recommended) or "2"
            save_checkpoints: Save perturbation after each iteration
            checkpoint_dir: Directory for checkpoints
            use_annotations: CRITICAL - Use real COCO captions for valid baseline (70%+)
            seed: Random seed for reproducibility
            
        Returns:
            Universal perturbation tensor (1, 3, 224, 224)
        """
        
        total_images = len(self.data_loader)
        
        print("="*60)
        print("UNIVERSAL ADVERSARIAL PERTURBATION GENERATION")
        print("Algorithm: Semantic Loss Minimization for CLIP")
        print("="*60)
        print(f"Training set:       {total_images:,} images (FULL dataset)")
        print(f"Iterations:         {num_iterations} full passes")
        print(f"Perturbation bound: {xi:.4f} (L-{norm_type})")
        print(f"Learning rate:      {learning_rate:.4f}")
        print(f"Batch size:         {batch_size}")
        print(f"Momentum:           {use_momentum} (decay={momentum_decay})")
        print(f"Alpha blending:     {alpha:.2f} (mobile app)")
        print(f"Target fooling:     {(1-delta)*100:.1f}%")
        
        # Check annotation status
        has_annotations = len(self.data_loader.image_to_captions) > 0
        print(f"Using annotations:  {use_annotations and has_annotations}")
        if use_annotations and not has_annotations:
            print("  WARNING: use_annotations=True but no annotations loaded!")
            print("  Falling back to generic descriptions")
        elif use_annotations and has_annotations:
            print(f"  ✓ Using {len(self.data_loader.image_to_captions)} human-verified COCO captions")
        
        print("="*60)
        
        # ── Phase 1: Load entire dataset once ──────────────────────
        all_image_paths = list(self.data_loader.image_paths)  # all 5K paths
        text_descriptions = self.data_loader.create_text_descriptions_for_batch(
            total_images,
            use_annotations=use_annotations,
            image_paths=all_image_paths
        )
        
        print(f"\n[Phase 1] Loading all {total_images:,} images (one-time cost) ...")
        dataset: List[torch.Tensor] = []
        valid_texts: List[str] = []
        for img_path, desc in tqdm(zip(all_image_paths, text_descriptions),
                                   total=total_images, desc="Loading", ncols=80):
            try:
                img_tensor = self.clip_model._load_image(img_path)
                dataset.append(img_tensor)
                valid_texts.append(desc)
            except Exception:
                continue
        
        actual_n = len(dataset)
        print(f"  Loaded {actual_n:,} / {total_images:,} images successfully")
        
        if actual_n == 0:
            raise RuntimeError("No valid images loaded — check data/MS-COCO/val2017")
        
        # ── Phase 2: Pre-compute text features once ────────────────
        print(f"[Phase 2] Pre-computing {actual_n:,} text embeddings (one-time) ...")
        text_features_list = []
        for desc in tqdm(valid_texts, desc="Text", ncols=80):
            text_emb = self.clip_model.get_text_embeddings(desc)
            text_features_list.append(torch.from_numpy(text_emb).to(self.device))
        text_features = torch.stack(text_features_list)
        
        # ── Phase 3: Compute baseline ──────────────────────────────
        print(f"[Phase 3] Computing baseline similarities ...")
        baseline_sim = self._compute_baseline(
            all_image_paths[:100],
            text_descriptions[:100] if use_annotations and has_annotations else None
        )
        print(f"  Baseline avg similarity: {baseline_sim:.2f}%")
        
        if baseline_sim < 50.0:
            print("\n⚠️  WARNING: Low baseline similarity detected!")
            print(f"   Current baseline: {baseline_sim:.2f}%")
            print(f"   Required: 70%+ for valid fooling rate measurement")
            print(f"   → Ensure use_annotations=True with captions_val2017.json")
            print(f"   → Generic descriptions like 'a photograph' produce low baselines")
        elif baseline_sim >= 70.0:
            print(f"  ✓ Valid baseline for fooling rate measurement (70%+ achieved)")
        
        # ── Phase 4: I-FGSM optimisation over full dataset ─────────
        print(f"\n[Phase 4] Optimizing perturbation over {actual_n:,} images × {num_iterations} iterations ...")
        print("-"*60)
        
        v = torch.zeros(1, 3, 224, 224, device=self.device)
        momentum = torch.zeros_like(v)
        
        history = {
            'iterations': [],
            'losses': [],
            'avg_similarities': [],
            'fooling_rates': []
        }
        
        # Build shuffled index order (reproducible)
        rng = np.random.RandomState(seed)
        
        for iteration in range(num_iterations):
            print(f"\n--- Iteration {iteration+1}/{num_iterations}  "
                  f"({actual_n:,} images) ---")
            
            # Shuffle traversal order each iteration for better generalisation
            order = rng.permutation(actual_n)
            
            epoch_loss = 0.0
            num_processed = 0
            
            for start in tqdm(range(0, actual_n, batch_size), desc=f"Iter {iteration+1}", ncols=80):
                batch_idx = order[start:start + batch_size]
                if len(batch_idx) == 0:
                    continue

                imgs = torch.cat([dataset[i] for i in batch_idx], dim=0)
                text_feat = text_features[batch_idx]

                # Create perturbation copy with gradient tracking
                v_grad = v.clone().detach().requires_grad_(True)

                # Apply alpha blending
                blended = self._apply_alpha_blending(imgs, v_grad, alpha)

                # Compute semantic loss
                loss = self._semantic_loss(blended, text_feat)

                # Backward pass
                loss.backward()

                # Update perturbation using MI-FGSM (momentum) or I-FGSM
                # Formula: v_{t+1} = Π_ε { v_t - α · sign(g_t) }
                with torch.no_grad():
                    grad = v_grad.grad
                    if use_momentum:
                        # Normalize to stabilize updates across batches
                        grad_norm = grad.abs().mean().clamp_min(1e-12)
                        momentum = momentum_decay * momentum + (grad / grad_norm)
                        step = momentum.sign()
                    else:
                        step = grad.sign()

                    # Gradient descent to minimize cosine similarity
                    v = v - learning_rate * step

                    # Project onto L_p ball (maintain perturbation bound)
                    v = self._project_perturbation(v, xi, norm_type)

                epoch_loss += loss.item() * len(batch_idx)
                num_processed += len(batch_idx)
            
            avg_loss = epoch_loss / num_processed if num_processed > 0 else 0
            
            # Evaluate on a representative subset (first 200 images)
            eval_n = min(200, actual_n)
            print(f"\nEvaluating on {eval_n} images ...")
            eval_results = self._evaluate(
                dataset[:eval_n], text_features[:eval_n], v, alpha, baseline_sim
            )
            
            # Store metrics
            history['iterations'].append(iteration + 1)
            history['losses'].append(avg_loss)
            history['avg_similarities'].append(eval_results['avg_similarity'])
            history['fooling_rates'].append(eval_results['fooling_rate'])
            
            # Print results
            print(f"\n{'='*60}")
            print(f"Iteration {iteration+1} Complete  "
                  f"(trained on {num_processed:,} images)")
            print(f"{'='*60}")
            print(f"Avg Loss:           {avg_loss:.4f}")
            print(f"Original Sim:       {baseline_sim:.2f}%")
            print(f"Protected Sim:      {eval_results['avg_similarity']:.2f}%")
            print(f"Similarity Drop:    {baseline_sim - eval_results['avg_similarity']:.2f}%")
            print(f"Fooling Rate:       {eval_results['fooling_rate']*100:.2f}%")
            print(f"Target:             {(1-delta)*100:.1f}%")
            print(f"Perturbation L-inf: {torch.max(torch.abs(v)).item():.6f}")
            print(f"{'='*60}")
            
            # Save checkpoint
            if save_checkpoints:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"uap_checkpoint_iter{iteration+1}.npy"
                )
                np.save(checkpoint_path, v.cpu().numpy())
                print(f"Saved checkpoint: {checkpoint_path}")
            
            # Check if target reached
            if eval_results['fooling_rate'] >= (1 - delta):
                print(f"\n✓ Target fooling rate achieved!")
                break
        
        # Final summary
        print(f"\n{'='*60}")
        print("UAP GENERATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Final iteration:    {iteration + 1}")
        print(f"Images per iter:    {actual_n:,} (full dataset)")
        print(f"Final fooling rate: {eval_results['fooling_rate']*100:.2f}%")
        print(f"Similarity drop:    {baseline_sim - eval_results['avg_similarity']:.2f}%")
        print(f"Perturbation norm:  {torch.max(torch.abs(v)).item():.6f}")
        print(f"{'='*60}")
        
        # Ensure output directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save final perturbation
        final_path = os.path.join(checkpoint_dir, "clip_uap_final.npy")
        np.save(final_path, v.cpu().numpy())
        print(f"\n✓ Saved final UAP: {final_path}")
        
        # Save training history
        self._save_history(history, checkpoint_dir)
        
        return v
    
    def _compute_baseline(self,
                          image_paths: List[str],
                          text_descriptions: Optional[List[str]] = None) -> float:
        """Compute average baseline similarity"""
        similarities = []
        use_texts = text_descriptions is not None

        for i, img_path in enumerate(image_paths[:100]):
            try:
                text = text_descriptions[i] if use_texts else "a photograph"
                sim = self.clip_model.compute_similarity(img_path, text)
                similarities.append(sim)
            except Exception:
                continue

        return np.mean(similarities) if similarities else 50.0
    
    def _evaluate(self, 
                  dataset: List[torch.Tensor],
                  text_features: torch.Tensor,
                  perturbation: torch.Tensor,
                  alpha: float,
                  baseline_sim: float) -> Dict:
        """Evaluate perturbation on dataset"""
        
        similarities = []
        fooled_count = 0
        
        with torch.no_grad():
            for idx in range(len(dataset)):
                img = dataset[idx]
                text_feat = text_features[idx:idx+1]
                
                # Apply perturbation
                blended = self._apply_alpha_blending(img, perturbation, alpha)
                
                # Compute similarity
                img_feat = self.model.encode_image(blended)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                sim = (100.0 * (img_feat * text_feat).sum()).item()
                
                similarities.append(sim)
                
                # Count as fooled if similarity dropped >30%
                if sim < baseline_sim * 0.7:
                    fooled_count += 1
        
        return {
            'avg_similarity': np.mean(similarities),
            'fooling_rate': fooled_count / len(dataset)
        }
    
    def _save_history(self, history: Dict, save_dir: str):
        """Save training history and plot"""
        
        # Save raw data
        history_path = os.path.join(save_dir, "training_history.npy")
        np.save(history_path, history)
        print(f"✓ Saved training history: {history_path}")

        # Save CSV for easy inspection
        csv_path = os.path.join(save_dir, "training_history.csv")
        with open(csv_path, "w", encoding="utf-8") as handle:
            handle.write("iteration,loss,avg_similarity,fooling_rate\n")
            for i in range(len(history["iterations"])):
                handle.write(
                    f"{history['iterations'][i]},"
                    f"{history['losses'][i]},"
                    f"{history['avg_similarities'][i]},"
                    f"{history['fooling_rates'][i]}\n"
                )
        print(f"✓ Saved training CSV: {csv_path}")
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(history['iterations'], history['losses'], 'b-o')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Similarity
        axes[0, 1].plot(history['iterations'], history['avg_similarities'], 'r-o')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Avg Similarity (%)')
        axes[0, 1].set_title('Protected Image Similarity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Fooling rate
        axes[1, 0].plot(history['iterations'], 
                       [r*100 for r in history['fooling_rates']], 'g-o')
        axes[1, 0].axhline(y=80, color='r', linestyle='--', label='Target 80%')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Fooling Rate (%)')
        axes[1, 0].set_title('Fooling Rate Progress')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary table
        axes[1, 1].axis('off')
        summary_text = f"""
        TRAINING SUMMARY
        
        Total Iterations: {len(history['iterations'])}
        
        Final Metrics:
        - Loss: {history['losses'][-1]:.4f}
        - Similarity: {history['avg_similarities'][-1]:.2f}%
        - Fooling Rate: {history['fooling_rates'][-1]*100:.2f}%
        
        Best Fooling Rate: {max(history['fooling_rates'])*100:.2f}%
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace')
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, "training_progress.png")
        plt.savefig(plot_path, dpi=150)
        print(f"✓ Saved training plot: {plot_path}")
        plt.close()


def main():
    """
    Main entry point for UAP generation
    """
    print("Initializing UAP Generator...")
    print("-"*60)
    
    # Define annotation path - resolve relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ann_path = os.path.join(script_dir, "..", "data", "MS-COCO", "annotations", "captions_val2017.json")
    ann_path = os.path.normpath(ann_path)
    
    # Initialize components
    print("\n[1/3] Loading CLIP model...")
    clip_model = CLIPModelWrapper(model_name="ViT-B/32")
    
    print(f"\n[2/3] Loading MS-COCO dataset...")
    print(f"  Annotations: {ann_path}")
    data_loader = COCODataLoader(annotations_file=ann_path)
    
    print("\n[3/3] Initializing generator...")
    generator = UniversalPerturbationGenerator(
        clip_model=clip_model,
        data_loader=data_loader
    )
    
    # Generate UAP
    print("\n" + "="*60)
    print("Starting UAP Generation")
    print("="*60)
    print("\nParameter Rationale:")
    print("  • xi=32/255: Stronger perturbation for measurable similarity drop")
    print("  • lr=8/255: Aggressive steps to overcome alpha-blending dilution")
    print("  • use_annotations=True: Ground-truth captions for 70%+ baseline")
    print("  • Full 5K dataset: Every iteration trains on all images")
    print("="*60)
    
    perturbation = generator.generate(
        num_iterations=10,    # 10 full passes over all 5K images
        xi=32/255,           # Increased perturbation bound for stronger effect
        learning_rate=8/255, # Increased step size for faster convergence
        alpha=0.7,           # Alpha blending for mobile
        delta=0.2,           # 80% fooling target
        norm_type="inf",     # L-infinity norm
        save_checkpoints=True,
        use_annotations=True,# CRITICAL: Use MS-COCO ground-truth captions for valid baseline
        seed=42              # Reproducibility
    )
    
    print("\n✓ UAP generation complete!")
    print("\nExpected Results with Optimized Parameters:")
    print("  • Baseline similarity: 70-85% (with ground-truth annotations)")
    print("  • Protected similarity: Should decrease to <30%")
    print("  • Fooling rate: Should reach 80% target")
    print("  • SSIM: Should remain >0.90 (imperceptible to humans)")
    print("\nNext steps:")
    print("  1. Run fidelity_validator.py to check SSIM/PSNR")
    print("  2. Test on new images")
    print("  3. Deploy to mobile app")


if __name__ == "__main__":
    main()
