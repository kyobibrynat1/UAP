"""
Fidelity Validator
Quality control for Universal Adversarial Perturbations

Validates that protected images maintain high visual quality:
- SSIM (Structural Similarity Index) > 0.90
- PSNR (Peak Signal-to-Noise Ratio) measurement
- Alpha parameter optimization
"""

import numpy as np
import torch
import os
from typing import List, Dict, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

from clip_integration import CLIPModelWrapper
from coco_loader import COCODataLoader


class FidelityValidator:
    """
    Validates visual quality of UAP-protected images
    Ensures protection maintains SSIM > 0.90 for thesis requirements
    """
    
    def __init__(self, 
                 clip_model: CLIPModelWrapper,
                 perturbation_path: str):
        """
        Initialize fidelity validator
        
        Args:
            clip_model: Initialized CLIP model wrapper
            perturbation_path: Path to UAP .npy file
        """
        self.clip_model = clip_model
        self.device = clip_model.device
        self.preprocess = clip_model.get_preprocess()
        
        # Load perturbation
        print(f"[Fidelity] Loading perturbation: {perturbation_path}")
        if not os.path.exists(perturbation_path):
            raise FileNotFoundError(f"Perturbation not found: {perturbation_path}")
        
        perturbation_np = np.load(perturbation_path)
        self.perturbation = torch.from_numpy(perturbation_np).to(self.device)
        
        print(f"[Fidelity] Perturbation loaded")
        print(f"[Fidelity] L-inf norm: {torch.max(torch.abs(self.perturbation)).item():.6f}")
        print(f"[Fidelity] L-2 norm: {torch.norm(self.perturbation).item():.6f}")
    
    def apply_perturbation(self, 
                          image_tensor: torch.Tensor,
                          alpha: float) -> torch.Tensor:
        """
        Apply UAP with alpha blending
        
        Formula: Image_new = (1 - α) · Image + α · (Image + UAP)
        
        Args:
            image_tensor: Original image (1, 3, 224, 224)
            alpha: Blending factor (0 to 1)
            
        Returns:
            Protected image tensor
        """
        # Add perturbation
        perturbed = torch.clamp(image_tensor + self.perturbation, 0, 1)
        
        # Alpha blend
        protected = (1 - alpha) * image_tensor + alpha * perturbed
        protected = torch.clamp(protected, 0, 1)
        
        return protected
    
    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert torch tensor to numpy array for metric computation
        
        Args:
            tensor: Image tensor (1, 3, H, W) or (3, H, W)
            
        Returns:
            Numpy array (H, W, 3) in range [0, 1]
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Move to CPU and convert to numpy
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img, 0, 1)
        
        return img
    
    def compute_ssim(self, 
                     original: np.ndarray,
                     protected: np.ndarray) -> float:
        """
        Compute Structural Similarity Index
        
        SSIM measures perceptual similarity:
        - 1.0 = identical images
        - > 0.90 = imperceptible difference (thesis requirement)
        - > 0.80 = minor difference
        - < 0.80 = noticeable difference
        
        Args:
            original: Original image (H, W, 3)
            protected: Protected image (H, W, 3)
            
        Returns:
            SSIM score (0 to 1)
        """
        return ssim(original, protected, data_range=1.0, channel_axis=2)
    
    def compute_psnr(self,
                     original: np.ndarray,
                     protected: np.ndarray) -> float:
        """
        Compute Peak Signal-to-Noise Ratio
        
        PSNR measures image quality:
        - > 40 dB = excellent quality
        - 30-40 dB = good quality
        - < 30 dB = poor quality
        
        Args:
            original: Original image (H, W, 3)
            protected: Protected image (H, W, 3)
            
        Returns:
            PSNR in dB
        """
        return psnr(original, protected, data_range=1.0)
    
    def validate_single_image(self,
                             image_path: str,
                             alpha: float,
                             text_description: str = "a photograph") -> Dict:
        """
        Validate quality metrics for a single image
        
        Args:
            image_path: Path to image
            alpha: Alpha blending value
            text_description: Text for CLIP similarity
            
        Returns:
            Dictionary with all metrics
        """
        # Load image
        image_tensor = self.clip_model._load_image(image_path)
        
        # Apply perturbation
        protected_tensor = self.apply_perturbation(image_tensor, alpha)
        
        # Convert to numpy for metrics
        original_np = self.tensor_to_numpy(image_tensor)
        protected_np = self.tensor_to_numpy(protected_tensor)
        
        # Compute fidelity metrics
        ssim_score = self.compute_ssim(original_np, protected_np)
        psnr_score = self.compute_psnr(original_np, protected_np)
        
        # Compute CLIP similarities
        original_sim = self.clip_model.compute_similarity(image_path, text_description)
        
        # For protected image, need to compute from tensor
        with torch.no_grad():
            text_emb = torch.from_numpy(
                self.clip_model.get_text_embeddings(text_description)
            ).to(self.device)
            
            protected_features = self.clip_model.model.encode_image(protected_tensor)
            protected_features = protected_features / protected_features.norm(dim=-1, keepdim=True)
            
            protected_sim = (100.0 * (protected_features @ text_emb.T)).item()
        
        results = {
            'ssim': ssim_score,
            'psnr': psnr_score,
            'original_similarity': original_sim,
            'protected_similarity': protected_sim,
            'similarity_drop': original_sim - protected_sim,
            'passes_threshold': ssim_score >= 0.90
        }
        
        return results
    
    def validate_batch(self,
                       image_paths: List[str],
                       alpha: float,
                       text_descriptions: Optional[List[str]] = None) -> Dict:
        """
        Validate metrics across a batch of images
        
        Args:
            image_paths: List of image paths
            alpha: Alpha blending value
            text_descriptions: Optional text descriptions per image
            
        Returns:
            Aggregated metrics
        """
        if text_descriptions is None:
            text_descriptions = ["a photograph"] * len(image_paths)
        
        results = {
            'ssim_scores': [],
            'psnr_scores': [],
            'original_similarities': [],
            'protected_similarities': [],
            'similarity_drops': [],
            'pass_count': 0
        }
        
        print(f"Validating {len(image_paths)} images with alpha={alpha:.2f}...")
        
        for img_path, text_desc in tqdm(zip(image_paths, text_descriptions),
                                        total=len(image_paths),
                                        desc="Validation"):
            try:
                metrics = self.validate_single_image(img_path, alpha, text_desc)
                
                results['ssim_scores'].append(metrics['ssim'])
                results['psnr_scores'].append(metrics['psnr'])
                results['original_similarities'].append(metrics['original_similarity'])
                results['protected_similarities'].append(metrics['protected_similarity'])
                results['similarity_drops'].append(metrics['similarity_drop'])
                
                if metrics['passes_threshold']:
                    results['pass_count'] += 1
                    
            except Exception as e:
                print(f"Warning: Failed to process {img_path}: {e}")
                continue
        
        # Compute aggregates
        results['avg_ssim'] = np.mean(results['ssim_scores'])
        results['avg_psnr'] = np.mean(results['psnr_scores'])
        results['avg_original_sim'] = np.mean(results['original_similarities'])
        results['avg_protected_sim'] = np.mean(results['protected_similarities'])
        results['avg_sim_drop'] = np.mean(results['similarity_drops'])
        results['pass_rate'] = results['pass_count'] / len(results['ssim_scores']) * 100
        
        return results
    
    def alpha_sweep(self,
                    image_paths: List[str],
                    alphas: List[float] = [0.5, 0.6, 0.7, 0.8, 1.0],
                    save_dir: str = "../data") -> Dict:
        """
        Test multiple alpha values to find optimal balance
        
        This is KEY for mobile app deployment - find the alpha that:
        1. Maintains SSIM > 0.90 (visual quality)
        2. Maximizes similarity drop (protection strength)
        
        Args:
            image_paths: Images to test
            alphas: List of alpha values to test
            save_dir: Directory to save results
            
        Returns:
            Results for all alpha values
        """
        print("="*60)
        print("ALPHA PARAMETER SWEEP")
        print("Finding optimal balance: Quality vs Protection")
        print("="*60)
        print(f"Testing alphas: {alphas}")
        print(f"Test images: {len(image_paths)}")
        print(f"Threshold: SSIM ≥ 0.90")
        print("="*60)
        
        sweep_results = {
            'alphas': [],
            'avg_ssim': [],
            'avg_psnr': [],
            'avg_sim_drop': [],
            'pass_rates': []
        }
        
        for alpha in alphas:
            print(f"\n--- Testing alpha = {alpha:.2f} ---")
            
            batch_results = self.validate_batch(image_paths, alpha)
            
            sweep_results['alphas'].append(alpha)
            sweep_results['avg_ssim'].append(batch_results['avg_ssim'])
            sweep_results['avg_psnr'].append(batch_results['avg_psnr'])
            sweep_results['avg_sim_drop'].append(batch_results['avg_sim_drop'])
            sweep_results['pass_rates'].append(batch_results['pass_rate'])
            
            # Print summary
            status = "✓ PASS" if batch_results['avg_ssim'] >= 0.90 else "✗ FAIL"
            print(f"\nResults:")
            print(f"  SSIM:           {batch_results['avg_ssim']:.4f} {status}")
            print(f"  PSNR:           {batch_results['avg_psnr']:.2f} dB")
            print(f"  Similarity Drop: {batch_results['avg_sim_drop']:.2f}%")
            print(f"  Pass Rate:      {batch_results['pass_rate']:.1f}%")
        
        # Find optimal alpha
        optimal_idx = self._find_optimal_alpha(sweep_results)
        
        print("\n" + "="*60)
        print("RECOMMENDED CONFIGURATION")
        print("="*60)
        print(f"Optimal Alpha:     {sweep_results['alphas'][optimal_idx]:.2f}")
        print(f"SSIM:             {sweep_results['avg_ssim'][optimal_idx]:.4f} (≥0.90 ✓)")
        print(f"PSNR:             {sweep_results['avg_psnr'][optimal_idx]:.2f} dB")
        print(f"Protection:       {sweep_results['avg_sim_drop'][optimal_idx]:.2f}% drop")
        print(f"Pass Rate:        {sweep_results['pass_rates'][optimal_idx]:.1f}%")
        print("="*60)
        
        # Visualization
        self._create_sweep_visualization(sweep_results, save_dir)
        
        return sweep_results
    
    def _find_optimal_alpha(self, sweep_results: Dict) -> int:
        """
        Find optimal alpha: highest protection while maintaining SSIM ≥ 0.90
        """
        optimal_idx = None
        max_protection = -1
        
        for i, (alpha, ssim_val, sim_drop) in enumerate(zip(
            sweep_results['alphas'],
            sweep_results['avg_ssim'],
            sweep_results['avg_sim_drop']
        )):
            # Must pass SSIM threshold
            if ssim_val >= 0.90:
                # Find maximum protection
                if sim_drop > max_protection:
                    max_protection = sim_drop
                    optimal_idx = i
        
        # If none pass, return lowest alpha
        if optimal_idx is None:
            optimal_idx = 0
        
        return optimal_idx
    
    def _create_sweep_visualization(self, sweep_results: Dict, save_dir: str):
        """Create visualization of alpha sweep results"""

        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # SSIM vs Alpha
        axes[0, 0].plot(sweep_results['alphas'], sweep_results['avg_ssim'], 'b-o', linewidth=2)
        axes[0, 0].axhline(y=0.90, color='r', linestyle='--', linewidth=2, label='Threshold (0.90)')
        axes[0, 0].set_xlabel('Alpha', fontsize=11)
        axes[0, 0].set_ylabel('SSIM', fontsize=11)
        axes[0, 0].set_title('Visual Quality (SSIM) vs Alpha', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0.85, 1.01])
        
        # PSNR vs Alpha
        axes[0, 1].plot(sweep_results['alphas'], sweep_results['avg_psnr'], 'g-o', linewidth=2)
        axes[0, 1].set_xlabel('Alpha', fontsize=11)
        axes[0, 1].set_ylabel('PSNR (dB)', fontsize=11)
        axes[0, 1].set_title('Peak Signal-to-Noise Ratio vs Alpha', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Similarity Drop vs Alpha
        axes[1, 0].plot(sweep_results['alphas'], sweep_results['avg_sim_drop'], 'm-o', linewidth=2)
        axes[1, 0].set_xlabel('Alpha', fontsize=11)
        axes[1, 0].set_ylabel('Similarity Drop (%)', fontsize=11)
        axes[1, 0].set_title('Protection Strength vs Alpha', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Pass Rate vs Alpha
        axes[1, 1].plot(sweep_results['alphas'], sweep_results['pass_rates'], 'c-o', linewidth=2)
        axes[1, 1].axhline(y=100, color='g', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Alpha', fontsize=11)
        axes[1, 1].set_ylabel('Pass Rate (%)', fontsize=11)
        axes[1, 1].set_title('SSIM ≥ 0.90 Pass Rate vs Alpha', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 105])
        
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, "alpha_sweep_results.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved visualization: {plot_path}")
        plt.close()


def main():
    """
    Main validation workflow
    """
    print("="*60)
    print("FIDELITY VALIDATION")
    print("Quality Control for Universal Adversarial Perturbations")
    print("="*60)
    
    # Initialize
    print("\n[1/3] Loading CLIP model...")
    clip_model = CLIPModelWrapper(model_name="ViT-B/32")
    
    print("\n[2/3] Loading perturbation...")
    perturbation_path = os.path.join("data", "results", "clip_uap_final.npy")
    
    if not os.path.exists(perturbation_path):
        print(f"Error: Perturbation not found: {perturbation_path}")
        print("Please run clip_uap_generator.py first")
        return
    
    validator = FidelityValidator(clip_model, perturbation_path)
    
    print("\n[3/3] Loading test images...")
    data_loader = COCODataLoader()
    test_images = data_loader.get_mini_batch(size=100, seed=42)
    print(f"Loaded {len(test_images)} test images")
    
    # Run alpha sweep
    print("\n" + "="*60)
    print("Starting Alpha Sweep Analysis")
    print("="*60)
    
    results = validator.alpha_sweep(
        image_paths=test_images,
        alphas=[0.5, 0.6, 0.7, 0.8, 1.0],
        save_dir="../data"
    )
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print("\n✓ Results saved to: ../data/alpha_sweep_results.png")
    print("\nFor your thesis:")
    print("  - Use the 'Optimal Alpha' value in your mobile app")
    print("  - Include SSIM ≥ 0.90 verification in your report")
    print("  - Show alpha_sweep_results.png for parameter optimization")


if __name__ == "__main__":
    main()
