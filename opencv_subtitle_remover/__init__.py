"""
Lightweight OpenCV-based Subtitle Remover

A standalone module for removing subtitles from videos using only OpenCV's Telea algorithm.
No Deep Learning dependencies required.
"""

__version__ = "1.0.0"
__all__ = ["SubtitleDetector", "SubtitleRemover", "create_mask", "is_video_or_image"]

from .detector import SubtitleDetector
from .remover import SubtitleRemover
from .utils import create_mask, is_video_or_image
