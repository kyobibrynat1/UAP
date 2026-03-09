"""
MS-COCO Data Loader
Efficient data pipeline for MS-COCO 2017 dataset

Handles loading image-caption pairs with memory-efficient mini-batching
for UAP generation on CPU/limited memory environments
"""

import os
import glob
import random
import json
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict
from pathlib import Path


class COCODataLoader:
    """
    Memory-efficient loader for MS-COCO 2017 dataset
    Supports random mini-batching without loading entire dataset into RAM
    """
    
    def __init__(self, 
                 image_dir: str = None,
                 annotations_file: str = None,
                 split: str = "val2017"):
        """
        Initialize COCO data loader
        
        Args:
            image_dir: Directory containing COCO images. If None, searches workspace
            annotations_file: Path to COCO annotations JSON. Optional for UAP generation
            split: Dataset split ('train2017', 'val2017')
        """
        self.split = split
        
        # Find image directory
        if image_dir is None:
            self.image_dir = self._find_image_directory()
        else:
            self.image_dir = image_dir
        
        print(f"[COCO] Image directory: {self.image_dir}")
        
        # Load image paths
        self.image_paths = self._load_image_paths()
        print(f"[COCO] Found {len(self.image_paths)} images")
        
        # Load annotations if available
        self.annotations = None
        self.image_to_captions = {}
        
        if annotations_file and os.path.exists(annotations_file):
            self._load_annotations(annotations_file)
            print(f"[COCO] Loaded annotations with {len(self.image_to_captions)} image-caption mappings")
        else:
            print(f"[COCO] No annotations loaded - using default descriptions")
    
    def _find_image_directory(self) -> str:
        """
        Auto-detect MS-COCO image directory in workspace
        """
        # Search patterns for data/MS-COCO structure
        search_paths = [
            "../data/MS-COCO/val2017",     # From python/ directory
            "../data/MS-COCO/train2017",
            "data/MS-COCO/val2017",        # From workspace root
            "data/MS-COCO/train2017",
            "../data/MS-COCO",             # Generic MS-COCO dir
            "data/MS-COCO",
            "../../val2017",               # Legacy paths
            "../../train2017",
            "../val2017",
            "../train2017",
            "../..",                       # Two levels up (workspace root)
            "..",                          # One level up (python dir)
            ".",                           # Current directory
        ]
        
        for search_path in search_paths:
            # Check if directory has COCO-style images (000000*.jpg)
            pattern = os.path.join(search_path, "000000*.jpg")
            matches = glob.glob(pattern)
            if len(matches) > 100:  # Found COCO images
                return os.path.abspath(search_path)
        
        # Fallback to data/MS-COCO/val2017
        fallback = os.path.abspath("../data/MS-COCO/val2017")
        if os.path.exists(fallback):
            return fallback
        return os.path.abspath("../..")
    
    def _load_image_paths(self) -> List[str]:
        """
        Load all image file paths
        """
        # COCO images follow pattern: 000000######.jpg
        pattern = os.path.join(self.image_dir, "000000*.jpg")
        image_paths = sorted(glob.glob(pattern))
        
        if len(image_paths) == 0:
            # Try alternate pattern (any .jpg)
            pattern = os.path.join(self.image_dir, "*.jpg")
            image_paths = sorted(glob.glob(pattern))
        
        return image_paths
    
    def _load_annotations(self, annotations_file: str):
        """
        Load COCO annotations JSON
        """
        try:
            with open(annotations_file, 'r') as f:
                data = json.load(f)
            
            # Build image_id to filename mapping
            id_to_filename = {img['id']: img['file_name'] for img in data['images']}
            
            # Build image to captions mapping
            for ann in data['annotations']:
                image_id = ann['image_id']
                caption = ann['caption']
                filename = id_to_filename.get(image_id)
                
                if filename:
                    if filename not in self.image_to_captions:
                        self.image_to_captions[filename] = []
                    self.image_to_captions[filename].append(caption)
            
            self.annotations = data
        except Exception as e:
            print(f"[COCO] Warning: Could not load annotations: {e}")
    
    def get_mini_batch(self, size: int = 5000, seed: Optional[int] = None) -> List[str]:
        """
        Get a random mini-batch of image paths
        
        This is the KEY FUNCTION for memory-efficient UAP generation:
        - Selects random subset without loading all images
        - Enables iterative optimization on limited hardware
        - Representative sampling from full dataset
        
        Args:
            size: Number of images to sample
            seed: Random seed for reproducibility
            
        Returns:
            List of image file paths
        """
        if seed is not None:
            random.seed(seed)
        
        # Ensure we don't request more than available
        batch_size = min(size, len(self.image_paths))
        
        # Random sampling without replacement
        batch_paths = random.sample(self.image_paths, batch_size)
        
        return batch_paths
    
    def get_image_with_caption(self, image_path: str) -> Tuple[str, str]:
        """
        Get image path with its caption
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (image_path, caption)
        """
        filename = os.path.basename(image_path)
        
        # Get caption from annotations if available
        if filename in self.image_to_captions:
            # Use first caption (COCO has 5 captions per image)
            caption = self.image_to_captions[filename][0]
        else:
            # Default generic caption
            caption = "a photograph"
        
        return image_path, caption
    
    def get_batch_with_captions(self, 
                                 size: int = 5000, 
                                 seed: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        Get mini-batch with corresponding captions
        
        Args:
            size: Batch size
            seed: Random seed
            
        Returns:
            List of (image_path, caption) tuples
        """
        image_paths = self.get_mini_batch(size, seed)
        batch = [self.get_image_with_caption(path) for path in image_paths]
        return batch
    
    def get_diverse_descriptions(self, num_descriptions: int = 10) -> List[str]:
        """
        Get diverse text descriptions for UAP generation
        Useful when annotations are not available
        
        Args:
            num_descriptions: Number of diverse descriptions to return
            
        Returns:
            List of text descriptions
        """
        descriptions = [
            "a photo of people",
            "a photo of animals",
            "a photo of vehicles",
            "a photo of food",
            "a photo of outdoor scenery",
            "a photo of indoor spaces",
            "a photo of sports activities",
            "a photo of nature",
            "a photo of urban scenes",
            "a photo of everyday objects",
            "a photo of buildings and architecture",
            "a photo of transportation",
            "a photo of electronics",
            "a photo of furniture",
            "a photo of clothing and fashion",
            "a photo of plants and flowers",
            "a photo of water and beaches",
            "a photo of mountains and landscapes",
            "a photo of cities and streets",
            "a photo of home interiors",
        ]
        
        return descriptions[:num_descriptions]
    
    def create_text_descriptions_for_batch(self, 
                                           batch_size: int,
                                           use_annotations: bool = False,
                                           image_paths: Optional[List[str]] = None) -> List[str]:
        """
        Create text descriptions matched to a batch of images
        
        Args:
            batch_size: Number of descriptions needed
            use_annotations: Whether to use real COCO captions
            image_paths: Optional list of image paths to match captions to
            
        Returns:
            List of text descriptions
        """
        if use_annotations and len(self.image_to_captions) > 0:
            # Use real captions matched to specific images
            if image_paths is not None:
                # Extract captions for the provided image paths
                descriptions = [self.get_image_with_caption(path)[1] for path in image_paths[:batch_size]]
            else:
                # Fallback: sample random batch (may not match images in generator)
                batch_paths = self.get_mini_batch(batch_size)
                descriptions = [self.get_image_with_caption(path)[1] for path in batch_paths]
        else:
            # Use diverse generic descriptions
            base_descriptions = self.get_diverse_descriptions(20)
            # Repeat to match batch size
            descriptions = (base_descriptions * ((batch_size // len(base_descriptions)) + 1))[:batch_size]
        
        return descriptions
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics
        
        Returns:
            Dictionary with dataset information
        """
        stats = {
            'total_images': len(self.image_paths),
            'image_directory': self.image_dir,
            'has_annotations': len(self.image_to_captions) > 0,
            'split': self.split,
        }
        
        if self.annotations:
            stats['total_captions'] = len(self.annotations['annotations'])
            stats['images_with_captions'] = len(self.image_to_captions)
        
        return stats
    
    def __len__(self) -> int:
        """Return total number of images"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> str:
        """Get image path by index"""
        return self.image_paths[idx]


def demo():
    """
    Demonstration of COCO data loader usage
    """
    print("="*60)
    print("MS-COCO Data Loader Demo")
    print("="*60)
    
    # Initialize loader
    print("\n--- Initializing Data Loader ---")
    loader = COCODataLoader()
    
    # Show statistics
    print("\n--- Dataset Statistics ---")
    stats = loader.get_statistics()
    for key, value in stats.items():
        print(f"{key:25s}: {value}")
    
    # Example 1: Get mini-batch
    print("\n--- Example 1: Mini-Batch Sampling ---")
    batch_size = 10
    batch = loader.get_mini_batch(size=batch_size, seed=42)
    print(f"Sampled {len(batch)} images:")
    for i, path in enumerate(batch[:5]):
        print(f"  {i+1}. {os.path.basename(path)}")
    print(f"  ... and {len(batch)-5} more")
    
    # Example 2: Get batch with captions
    print("\n--- Example 2: Batch with Captions ---")
    batch_with_captions = loader.get_batch_with_captions(size=5, seed=42)
    for i, (path, caption) in enumerate(batch_with_captions):
        print(f"  {i+1}. {os.path.basename(path)}")
        print(f"     Caption: {caption[:60]}...")
    
    # Example 3: Diverse descriptions
    print("\n--- Example 3: Diverse Descriptions ---")
    descriptions = loader.get_diverse_descriptions(num_descriptions=5)
    for i, desc in enumerate(descriptions):
        print(f"  {i+1}. {desc}")
    
    # Example 4: Memory-efficient UAP workflow
    print("\n--- Example 4: UAP Generation Workflow ---")
    print("Typical usage for UAP generation:")
    print(f"  1. Total dataset: {len(loader)} images")
    print(f"  2. Mini-batch size: 5000 images (memory efficient)")
    print(f"  3. Number of batches: {len(loader) // 5000}")
    print(f"  4. Can iterate multiple passes over different batches")
    
    # Show how to use in UAP generation
    print("\nCode example:")
    print("```python")
    print("loader = COCODataLoader()")
    print("for pass_num in range(5):  # Multiple passes")
    print("    batch = loader.get_mini_batch(size=5000)")
    print("    descriptions = loader.create_text_descriptions_for_batch(5000)")
    print("    # Run UAP optimization on this batch")
    print("    # perturbation = optimize(batch, descriptions)")
    print("```")
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nKey Function: loader.get_mini_batch(size=5000)")
    print("  - Randomly samples images without loading into memory")
    print("  - Enables UAP generation on CPU with limited RAM")
    print("  - Representative sampling from 100K+ images")


if __name__ == "__main__":
    demo()
