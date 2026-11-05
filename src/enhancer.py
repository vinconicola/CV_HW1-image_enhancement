import cv2
import numpy as np
import os
from pathlib import Path

def apply_enhancement_methods(input_dir, output_dir):
    """
    Apply multiple enhancement methods to all images in the input directory,
    and save the results in separate folders based on the method applied.

    Args:
        input_dir (str): Directory containing the original or degraded images.
        output_dir (str): Base directory to save the enhanced images.
    """
    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # List of enhancement methods and their corresponding output subdirectories
    methods = {
        "linear_contrast_stretching": "linear_stretching",
        "histogram_equalization": "histogram_eq",
        "gamma_correction": "gamma"
    }

    # Create subdirectories for each method
    for method_name, method_folder in methods.items():
        method_output_dir = os.path.join(output_dir, method_folder)
        Path(method_output_dir).mkdir(parents=True, exist_ok=True)

    # Get list of image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # Apply each enhancement method to all images
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)

        # Apply linear contrast stretching
        linear_output = os.path.join(output_dir, methods["linear_contrast_stretching"], image_file)
        linear_contrast_stretching(image_path, linear_output)

        # Apply histogram equalization
        histogram_output = os.path.join(output_dir, methods["histogram_equalization"], image_file)
        histogram_equalization(image_path, histogram_output)

        # Apply gamma correction
        gamma_output = os.path.join(output_dir, methods["gamma_correction"], image_file)
        gamma_correction(image_path, gamma_output)

    print(f"Enhancement methods applied to {len(image_files)} images and saved to {output_dir}.")


def linear_contrast_stretching(image_path, output_path):
    """
    Apply luminance-based linear contrast stretching to the image
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")

    # Convert image to CIELAB colorspace
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Extract L (lightness), A, B channels
    l_channel = lab_image[:, :, 0]
    a_channel = lab_image[:, :, 1]
    b_channel = lab_image[:, :, 2]

    # Apply linear contrast stretching on L channel
    l_min = np.min(l_channel)
    l_max = np.max(l_channel)

    # Avoid division by zero
    if l_max == l_min:
        stretched_l = l_channel
    else:
        stretched_l = ((l_channel - l_min) / (l_max - l_min)) * 255

    # Convert stretched L back to 8-bit unsigned integer
    stretched_l = np.clip(stretched_l, 0, 255).astype(np.uint8)

    # Recombine the L, A, B channels
    enhanced_lab = np.stack([stretched_l, a_channel, b_channel], axis=-1)

    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Save the enhanced image
    cv2.imwrite(output_path, enhanced_image)


def histogram_equalization(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")

    # Convert image to HSV colorspace
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract H (hue), S (saturation), V (value) channels
    h_channel = hsv_image[:, :, 0]
    s_channel = hsv_image[:, :, 1]
    v_channel = hsv_image[:, :, 2]

    # Apply histogram equalization to the V channel
    enhanced_v = cv2.equalizeHist(v_channel)

    # Recombine the H, S, enhanced V channels
    enhanced_hsv = np.stack([h_channel, s_channel, enhanced_v], axis=-1)

    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    # Save the enhanced image
    cv2.imwrite(output_path, enhanced_image)


def gamma_correction(image_path, output_path):
    """
    Apply gamma correction to the image
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")

    # Convert image to float for processing
    image = np.float32(image) / 255.0

    # Define gamma value
    gamma = 1.2 

    # Apply gamma correction
    enhanced_image = np.power(image, gamma)

    # Convert back to 8-bit unsigned integer
    enhanced_image = np.uint8(enhanced_image * 255)

    # Save the enhanced image
    cv2.imwrite(output_path, enhanced_image)


if __name__ == "__main__":
    # Define input and output directories
    input_dir = "images/degraded"
    output_dir = "images/enhanced"

    # Apply enhancement methods to all images in the input directory
    apply_enhancement_methods(input_dir, output_dir)