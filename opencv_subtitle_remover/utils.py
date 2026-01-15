"""
Utility functions for the OpenCV subtitle remover module.
"""

import os
import numpy as np


def is_video_or_image(file_path):
    """
    Check if the file is a video or image based on extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if file is a video or image, False otherwise
    """
    video_ext = {'.mp4', '.avi', '.mkv', '.flv', '.mov', '.wmv', '.webm', '.m4v', '.mpg', '.mpeg'}
    image_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    if not os.path.exists(file_path):
        return False
    
    ext = os.path.splitext(file_path)[-1].lower()
    return ext in video_ext or ext in image_ext


def create_mask(mask_size, coordinates_list):
    """
    Create a binary mask from a list of coordinates.
    
    Args:
        mask_size: Tuple of (height, width) for the mask
        coordinates_list: List of (xmin, xmax, ymin, ymax) tuples
        
    Returns:
        numpy.ndarray: Binary mask (uint8)
    """
    import cv2
    mask = np.zeros(mask_size, dtype=np.uint8)
    
    for coords in coordinates_list:
        xmin, xmax, ymin, ymax = coords
        # Use cv2.rectangle for proper mask creation (matches backend behavior)
        cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), (255, 255, 255), thickness=-1)
    
    return mask


def expand_coordinates(coords, expansion_pixels=20):
    """
    Expand coordinate boundaries by a specified number of pixels.
    
    Args:
        coords: Tuple of (xmin, xmax, ymin, ymax)
        expansion_pixels: Number of pixels to expand in each direction
        
    Returns:
        Tuple of expanded (xmin, xmax, ymin, ymax)
    """
    xmin, xmax, ymin, ymax = coords
    return (
        max(0, xmin - expansion_pixels),
        xmax + expansion_pixels,
        max(0, ymin - expansion_pixels),
        ymax + expansion_pixels
    )
