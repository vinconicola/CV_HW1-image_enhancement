import numpy as np
import cv2
from pathlib import Path

#Create a degraded version of an image by reducing contrast and adding Gaussian noise
def degrade_image(image_path: str, 
                  output_dir: str = "images/degraded",
                  contrast_reduction: float = 0.15,  # Reduce intensity by this factor (e.g., 0.15 = reduce by 15%)
                  noise_std: float = 10,             # Standard deviation of Gaussian noise (in pixel values)
                  random_seed: int = 42):            # Fixed seed for reproducibility
    
    

    
    
    # Convert paths to Path objects for handling
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    original_image = cv2.imread(str(image_path))
    
    if original_image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # Get image properties
    height, width, _ = original_image.shape

    # Convert to float for processing (avoid overflow issues)
    image_float = original_image.astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Compress the range proportionally
    min_intensity = np.min(image_float)
    max_intensity = np.max(image_float)
    intensity_range = max_intensity - min_intensity
    compressed_range = intensity_range * (1 - contrast_reduction)
    
    # Create a new image with reduced range centered on the original mean
    image_mean = np.mean(image_float, axis=(0, 1), keepdims=True)
    normalized_image = (image_float - image_mean) / intensity_range
    
    # Compress the dynamic range
    compressed_normalized = normalized_image * compressed_range
    
    # Rescale back to original range
    degraded_contrast = (compressed_normalized + image_mean)
    
    # Generate noise in the same scale as the image
    np.random.seed(random_seed)
    noise = np.random.normal(0, noise_std / 255.0, degraded_contrast.shape)
    
    # Combine degraded image with noise
    degraded_image_float = degraded_contrast + noise
    
    # Clip to valid range [0, 1]
    degraded_image_float = np.clip(degraded_image_float, 0, 1)
    
    # Convert back to original dtype
    if original_image.dtype == np.uint8:
        degraded_image_uint8 = (degraded_image_float * 255).astype(np.uint8)
    else:
        degraded_image_uint8 = (degraded_image_float / degraded_image_float.max() * 255).astype(np.uint8)
    
    # Save the degraded image
    base_name = Path(image_path).stem
    degraded_image_path = output_dir / f"{base_name}_degraded.png"
    cv2.imwrite(str(degraded_image_path), degraded_image_uint8)
    
    # Create degradation parameters dictionary for reproducibility
    degradation_params = {
        "image_path": str(image_path),
        "output_path": str(degraded_image_path),
        "contrast_reduction": contrast_reduction,
        "noise_std": noise_std,
        "random_seed": random_seed,
        "original_shape": (height, width, 3),
        "original_dtype": original_image.dtype.itemsize * 8
    }
    
    return degraded_image_path, degradation_params

#Apply degradation to multiple images in a directory.
def degrade_multiple_images(image_dir: str = "images/originals", 
                           output_dir: str = "images/degraded",
                           contrast_reduction: float = 0.15,
                           noise_std: float = 10,
                           random_seed: int = 42):
    
    # Get all image files in the directory
    import os
    from glob import glob
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(image_dir, ext)))
    
    results = []
    for image_path in sorted(image_files):  # Sort for consistent ordering
        try:
            degraded_path, params = degrade_image(
                image_path=image_path,
                output_dir=output_dir,
                contrast_reduction=contrast_reduction,
                noise_std=noise_std,
                random_seed=random_seed
            )
            results.append((degraded_path, params))
            print(f"Degraded: {Path(image_path).name} -> {Path(degraded_path).name}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    return results


if __name__ == "__main__":
    # Example usage when run directly
    print("Running image degradation pipeline...")
    
    # Default parameters (feel free to adjust)
    CONTRAST_REDUCTION = 0.15  # Reduce contrast by 15%
    NOISE_STD = 10              # Standard deviation of noise
    RANDOM_SEED = 42            # For reproducibility
    
    # Process all images in the originals folder
    degradation_results = degrade_multiple_images(
        image_dir="images/originals",
        output_dir="images/degraded",
        contrast_reduction=CONTRAST_REDUCTION,
        noise_std=NOISE_STD,
        random_seed=RANDOM_SEED
    )
    
    print(f"\nTotal images processed: {len(degradation_results)}")
    print("\nDegradation parameters used:")
    print(f"  - Contrast reduction: {CONTRAST_REDUCTION} (reduce by {CONTRAST_REDUCTION*100:.0f}% of range)")
    print(f"  - Noise std dev: {NOISE_STD}")
    print(f"  - Random seed: {RANDOM_SEED}")