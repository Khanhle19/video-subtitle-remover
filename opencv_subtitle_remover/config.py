"""
Configuration for OpenCV subtitle remover.
Matches backend/config.py settings for consistent results.
"""

# Pixel tolerance settings
PIXEL_TOLERANCE_X = 20  # Horizontal deviation pixels for similar region detection
PIXEL_TOLERANCE_Y = 20  # Vertical deviation pixels for similar region detection

# Mask expansion settings
SUBTITLE_AREA_DEVIATION_PIXEL = 20  # Pixels to expand mask to avoid edge artifacts

# Inpaint settings
INPAINT_RADIUS = 3  # Radius for Telea inpainting algorithm
