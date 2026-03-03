"""
CLIP Universal Adversarial Perturbation Generator
Core algorithm for generating semantic UAPs against CLIP ViT-B/32

This is the MAIN algorithm that creates the "Universal Cloak" by:
1. Minimizing cosine similarity between images and text descriptions
2. Using alpha blending for mobile app compatibility
3. Memory-efficient mini-batch processing
"""

import numpy as np
import torch
import clip
import os
import random
import contextlib
from tqdm import tqdm
from typing import Optional, Dict, List
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

        # Freeze CLIP weights to save memory; we only need gradients on the perturbation
        for param in self.model.parameters():
            param.requires_grad_(False)
        
        print("[UAP] Generator initialized")
    
    def _semantic_loss(self, image_tensor: torch.Tensor,
                       text_features: torch.Tensor,
                       negative_text_features: Optional[torch.Tensor] = None,
                       classification_weight: float = 0.0) -> torch.Tensor:
        """
        Compute semantic loss: cosine similarity between image and text
        
        Goal: MINIMIZE this loss to "cloak" the image from AI understanding
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

        # Compute cosine similarity with matched captions
        similarity = (image_features * text_features).sum(dim=1)
        loss = similarity.mean()

        # Optional classification-style loss using negative prompts
        if negative_text_features is not None and classification_weight > 0:
            neg_sims = image_features @ negative_text_features.T
            loss = loss - classification_weight * neg_sims.mean()

        return loss
    
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

    def _get_text_features_for_paths(self,
                                     image_paths: List[str],
                                     use_annotations: bool,
                                     captions_per_image: int = 1,
                                     seed: Optional[int] = None) -> torch.Tensor:
        """
        Build text features for a list of image paths.

        If annotations are available, sample captions per image and average their embeddings.
        Otherwise, use diverse generic descriptions.
        """
        rng = random.Random(seed) if seed is not None else random
        captions_per_image = max(1, int(captions_per_image))

        # Build list-of-lists of captions per image
        descriptions: List[List[str]] = []
        if use_annotations and len(self.data_loader.image_to_captions) > 0:
            for img_path in image_paths:
                filename = os.path.basename(img_path)
                captions = self.data_loader.image_to_captions.get(filename, [])
                if captions:
                    if captions_per_image > 1:
                        k = min(captions_per_image, len(captions))
                        chosen = rng.sample(captions, k=k)
                        descriptions.append(chosen)
                    else:
                        descriptions.append([rng.choice(captions)])
                else:
                    descriptions.append(["a photograph"])
        else:
            base_descriptions = self.data_loader.get_diverse_descriptions(20)
            for idx in range(len(image_paths)):
                descriptions.append([base_descriptions[idx % len(base_descriptions)]])

        # Flatten captions for batch embedding
        flat_captions = [cap for caps in descriptions for cap in caps]
        text_embs = self.clip_model.get_text_embeddings(flat_captions)
        text_embs = np.atleast_2d(text_embs)

        # Aggregate per image (mean + renorm)
        text_features = []
        cursor = 0
        for caps in descriptions:
            cap_count = len(caps)
            emb = text_embs[cursor:cursor + cap_count].mean(axis=0)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            text_features.append(torch.from_numpy(emb).to(self.device))
            cursor += cap_count

        return torch.stack(text_features)
    
    def generate(self,
                 batch_size: int = 1000,
                 micro_batch_size: int = 16,
                 num_iterations: int = 10,
                 xi: float = 16/255,
                 learning_rate: float = 2/255,
                 alpha: float = 0.7,
                 delta: float = 0.2,
                 fooling_drop_ratio: float = 0.7,
                 classification_weight: float = 0.0,
                 negative_prompt_pool_size: int = 20,
                 negative_prompts_per_batch: int = 4,
                 norm_type: str = "inf",
                 save_checkpoints: bool = True,
                 checkpoint_dir: str = "data/results",
                 use_annotations: bool = False,
                 captions_per_image: int = 1,
                 eval_sample_size: int = 100,
                 eval_micro_batch_size: int = 16,
                 use_amp: bool = False,
                 seed: Optional[int] = None) -> torch.Tensor:
        """
        Generate Universal Adversarial Perturbation
        
        KEY ALGORITHM - This is where the magic happens!
        
        Args:
            batch_size: Number of images per mini-batch (1000 for CPU efficiency)
            micro_batch_size: Images per GPU micro-batch (VRAM safety)
            num_iterations: Number of optimization passes
            xi: Perturbation magnitude bound (16/255 ≈ 0.063)
            learning_rate: Gradient descent step size
            alpha: Alpha blending for mobile app (0.7 = 70% blend)
            delta: Target fooling rate (0.2 = 80% fooling)
            fooling_drop_ratio: Similarity drop ratio to count as fooled
            classification_weight: Weight for negative-prompt loss term
            negative_prompt_pool_size: Number of generic prompts to sample from
            negative_prompts_per_batch: Number of negative prompts per batch
            norm_type: "inf" (recommended) or "2"
            save_checkpoints: Save perturbation after each iteration
            checkpoint_dir: Directory for checkpoints
            use_annotations: Use real COCO captions (requires annotations file)
            captions_per_image: Number of captions per image to average (if annotations)
            eval_sample_size: Number of images to evaluate per iteration
            eval_micro_batch_size: Micro-batch size for evaluation
            use_amp: Enable CUDA autocast for lower VRAM usage
            seed: Random seed for reproducibility
            
        Returns:
            Universal perturbation tensor (1, 3, 224, 224)
        """
        
        print("="*60)
        print("UNIVERSAL ADVERSARIAL PERTURBATION GENERATION")
        print("Algorithm: Semantic Loss Minimization for CLIP")
        print("="*60)
        print(f"Dataset size:       {len(self.data_loader):,} images")
        print(f"Mini-batch size:    {batch_size:,} images")
        print(f"Micro-batch size:   {micro_batch_size:,} images")
        print(f"Iterations:         {num_iterations}")
        print(f"Perturbation bound: {xi:.4f} (L-{norm_type})")
        print(f"Learning rate:      {learning_rate:.4f}")
        print(f"Alpha blending:     {alpha:.2f} (mobile app)")
        print(f"Target fooling:     {(1-delta)*100:.1f}%")
        print(f"Fooling drop ratio: {fooling_drop_ratio:.2f}")
        print(f"Classif. weight:    {classification_weight:.2f}")
        print(f"Captions/image:     {captions_per_image}")
        print(f"Eval sample size:   {eval_sample_size}")
        
        # Check annotation status
        has_annotations = len(self.data_loader.image_to_captions) > 0
        print(f"Using annotations:  {use_annotations and has_annotations}")
        if use_annotations and not has_annotations:
            print("  WARNING: use_annotations=True but no annotations loaded!")
            print("  Falling back to generic descriptions")
        elif use_annotations and has_annotations:
            print(f"  ✓ Using {len(self.data_loader.image_to_captions)} human-verified COCO captions")
        
        print("="*60)
        
        # Initialize perturbation
        v = torch.zeros(1, 3, 224, 224, device=self.device)
        
        # Track metrics
        history = {
            'iterations': [],
            'losses': [],
            'avg_similarities': [],
            'fooling_rates': []
        }
        
        # Compute baseline (caption-matched)
        print("\n[Phase 1] Computing baseline similarities...")
        baseline_batch = self.data_loader.get_mini_batch(min(batch_size, 500), seed=seed)
        baseline_paths = baseline_batch[:min(eval_sample_size, len(baseline_batch))]
        baseline_sim = self._compute_baseline_on_paths(
            baseline_paths,
            use_annotations=use_annotations,
            captions_per_image=captions_per_image,
            micro_batch_size=eval_micro_batch_size
        )
        print(f"Baseline avg similarity: {baseline_sim:.2f}%")
        
        # Main optimization loop
        print(f"\n[Phase 2] Optimizing perturbation...")
        print("-"*60)
        
        for iteration in range(num_iterations):
            print(f"\n--- Iteration {iteration+1}/{num_iterations} ---")
            
            # Get mini-batch
            batch_seed = seed + iteration if seed is not None else None
            image_paths = self.data_loader.get_mini_batch(batch_size, seed=batch_seed)

            # Prepare negative prompt pool (shared for this iteration)
            neg_prompt_pool = None
            if classification_weight > 0:
                neg_prompts = self.data_loader.get_diverse_descriptions(negative_prompt_pool_size)
                neg_embs = self.clip_model.get_text_embeddings(neg_prompts)
                neg_embs = np.atleast_2d(neg_embs)
                neg_prompt_pool = torch.from_numpy(neg_embs).to(self.device)

            # Optimization loop over batch with micro-batching
            epoch_loss = 0.0
            num_processed = 0

            print("Optimizing...")
            v = v.detach().requires_grad_(True)
            if v.grad is not None:
                v.grad.zero_()

            amp_enabled = use_amp and self.device.startswith("cuda")
            rng = random.Random(batch_seed) if batch_seed is not None else random

            for start in tqdm(range(0, len(image_paths), micro_batch_size), desc="Optimize", ncols=80):
                chunk_paths = image_paths[start:start + micro_batch_size]
                images = []
                valid_paths = []

                for img_path in chunk_paths:
                    try:
                        img_tensor = self.clip_model._load_image(img_path)
                        images.append(img_tensor)
                        valid_paths.append(img_path)
                    except Exception:
                        continue

                if len(images) == 0:
                    continue

                image_batch = torch.cat(images, dim=0)
                text_features = self._get_text_features_for_paths(
                    valid_paths,
                    use_annotations=use_annotations,
                    captions_per_image=captions_per_image,
                    seed=batch_seed
                )

                neg_features = None
                if neg_prompt_pool is not None and negative_prompts_per_batch > 0:
                    k = min(negative_prompts_per_batch, neg_prompt_pool.size(0))
                    neg_idx = rng.sample(range(neg_prompt_pool.size(0)), k=k)
                    neg_features = neg_prompt_pool[neg_idx]
                    neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)

                with (torch.cuda.amp.autocast(enabled=True) if amp_enabled else contextlib.nullcontext()):
                    blended = self._apply_alpha_blending(image_batch, v, alpha)
                    loss = self._semantic_loss(
                        blended,
                        text_features,
                        negative_text_features=neg_features,
                        classification_weight=classification_weight
                    )

                loss.backward()

                epoch_loss += loss.item() * image_batch.size(0)
                num_processed += image_batch.size(0)

            if num_processed == 0 or v.grad is None:
                print("Warning: No valid images in batch, skipping...")
                continue

            # Update perturbation (gradient descent to minimize similarity)
            with torch.no_grad():
                v = v - learning_rate * v.grad.sign()
                v = self._project_perturbation(v, xi, norm_type)

            avg_loss = epoch_loss / num_processed

            # Evaluate
            print("\nEvaluating perturbation...")
            eval_paths = image_paths[:min(eval_sample_size, len(image_paths))]
            eval_results = self._evaluate_on_paths(
                eval_paths,
                use_annotations=use_annotations,
                captions_per_image=captions_per_image,
                perturbation=v,
                alpha=alpha,
                baseline_sim=baseline_sim,
                fooling_drop_ratio=fooling_drop_ratio,
                micro_batch_size=eval_micro_batch_size
            )
            
            # Store metrics
            history['iterations'].append(iteration + 1)
            history['losses'].append(avg_loss)
            history['avg_similarities'].append(eval_results['avg_similarity'])
            history['fooling_rates'].append(eval_results['fooling_rate'])
            
            # Print results
            print(f"\n{'='*60}")
            print(f"Iteration {iteration+1} Complete")
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
                # Ensure checkpoint directory exists
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
    
    def _compute_baseline_on_paths(self,
                                   image_paths: List[str],
                                   use_annotations: bool,
                                   captions_per_image: int,
                                   micro_batch_size: int = 16) -> float:
        """Compute average baseline similarity with the same captions as evaluation"""
        similarities = []

        with torch.no_grad():
            for start in range(0, len(image_paths), micro_batch_size):
                chunk_paths = image_paths[start:start + micro_batch_size]
                images = []
                valid_paths = []

                for img_path in chunk_paths:
                    try:
                        img_tensor = self.clip_model._load_image(img_path)
                        images.append(img_tensor)
                        valid_paths.append(img_path)
                    except Exception:
                        continue

                if len(images) == 0:
                    continue

                image_batch = torch.cat(images, dim=0)
                text_features = self._get_text_features_for_paths(
                    valid_paths,
                    use_annotations=use_annotations,
                    captions_per_image=captions_per_image
                )

                img_feat = self.model.encode_image(image_batch)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                sim = (100.0 * (img_feat * text_features).sum(dim=1)).detach().cpu().tolist()
                similarities.extend(sim)

        return np.mean(similarities) if similarities else 50.0
    
    def _evaluate_on_paths(self,
                           image_paths: List[str],
                           use_annotations: bool,
                           captions_per_image: int,
                           perturbation: torch.Tensor,
                           alpha: float,
                           baseline_sim: float,
                           fooling_drop_ratio: float = 0.7,
                           micro_batch_size: int = 16) -> Dict:
        """Evaluate perturbation on a list of image paths"""
        similarities = []
        fooled_count = 0

        with torch.no_grad():
            for start in range(0, len(image_paths), micro_batch_size):
                chunk_paths = image_paths[start:start + micro_batch_size]
                images = []
                valid_paths = []

                for img_path in chunk_paths:
                    try:
                        img_tensor = self.clip_model._load_image(img_path)
                        images.append(img_tensor)
                        valid_paths.append(img_path)
                    except Exception:
                        continue

                if len(images) == 0:
                    continue

                image_batch = torch.cat(images, dim=0)
                text_features = self._get_text_features_for_paths(
                    valid_paths,
                    use_annotations=use_annotations,
                    captions_per_image=captions_per_image
                )

                blended = self._apply_alpha_blending(image_batch, perturbation, alpha)
                img_feat = self.model.encode_image(blended)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                sim = (100.0 * (img_feat * text_features).sum(dim=1)).detach().cpu().tolist()

                similarities.extend(sim)
                fooled_count += sum(1 for s in sim if s < baseline_sim * fooling_drop_ratio)

        if len(similarities) == 0:
            return {
                'avg_similarity': baseline_sim,
                'fooling_rate': 0.0
            }

        return {
            'avg_similarity': np.mean(similarities),
            'fooling_rate': fooled_count / len(similarities)
        }
    
    def _save_history(self, history: Dict, save_dir: str):
        """Save training history and plot"""
        
        # Save raw data
        history_path = os.path.join(save_dir, "training_history.npy")
        np.save(history_path, history)
        print(f"✓ Saved training history: {history_path}")
        
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
    
    perturbation = generator.generate(
        batch_size=256,              # Images per iteration
        micro_batch_size=16,         # VRAM-safe micro-batch size
        num_iterations=12,           # More passes for stronger UAP
        xi=18/255,                   # Slightly higher noise allowance
        learning_rate=2/255,         # Step size
        alpha=0.75,                  # Slightly stronger blend
        delta=0.2,                   # 80% fooling target
        fooling_drop_ratio=0.80,     # Relaxed similarity drop threshold
        classification_weight=0.30,  # Negative-prompt loss weight
        negative_prompt_pool_size=20,# Prompt pool size
        negative_prompts_per_batch=4,# Prompts per batch
        norm_type="inf",             # L-infinity norm
        save_checkpoints=True,
        use_annotations=True,        # Use real COCO captions for training
        captions_per_image=2,        # Caption diversity improves robustness
        eval_sample_size=100,        # Evaluation sample size
        eval_micro_batch_size=16,    # VRAM-safe eval batch
        use_amp=False,               # Set True on CUDA if you want lower VRAM
        seed=42                      # Reproducibility
    )
    
    print("\n✓ UAP generation complete!")
    print("Next steps:")
    print("  1. Run fidelity_validator.py to check SSIM/PSNR")
    print("  2. Test on new images")
    print("  3. Deploy to mobile app")


if __name__ == "__main__":
    main()
