"""
Subtitle detection module using PaddleOCR.

This module provides lightweight subtitle detection without Deep Learning inpainting models.
"""

import os
import sys
import cv2
from functools import cached_property
from tqdm import tqdm

# Disable OneDNN before importing PaddleOCR to avoid compatibility issues
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from . import config
from .utils import expand_coordinates


class SubtitleDetector:
    """
    Lightweight subtitle detector using PaddleOCR text detection.
    """
    
    def __init__(self, video_path, sub_area=None):
        """
        Initialize subtitle detector.
        
        Args:
            video_path: Path to video file
            sub_area: Optional tuple (ymin, ymax, xmin, xmax) to restrict detection area
        """
        self.video_path = video_path
        self.sub_area = sub_area
    
    @cached_property
    def text_detector(self):
        """Lazy-load PaddleOCR text detector using public API."""
        from paddleocr import PaddleOCR
        
        # Use PaddleOCR's public API - minimal configuration
        ocr = PaddleOCR(
            lang='ch'
        )
        return ocr
    
    def detect_subtitle(self, img):
        """
        Detect text boxes in a single image.
        
        Args:
            img: Image array (BGR format)
            
        Returns:
            List of detection boxes
        """
        # Call OCR - it will do both detection and recognition
        result = self.text_detector.ocr(img)
        
        # Extract just the boxes from the result
        # Result format: [[box, (text, confidence)], ...]
        if result and result[0]:
            boxes = [line[0] for line in result[0]]
            return boxes
        return []
    
    @staticmethod
    def are_similar(region1, region2, tolerance_x=None, tolerance_y=None):
        """
        Determine if two regions are similar.
        
        Args:
            region1: Tuple of (xmin, xmax, ymin, ymax)
            region2: Tuple of (xmin, xmax, ymin, ymax)
            tolerance_x: Horizontal pixel tolerance (defaults to config)
            tolerance_y: Vertical pixel tolerance (defaults to config)
            
        Returns:
            bool: True if regions are similar
        """
        if tolerance_x is None:
            tolerance_x = config.PIXEL_TOLERANCE_X
        if tolerance_y is None:
            tolerance_y = config.PIXEL_TOLERANCE_Y
            
        xmin1, xmax1, ymin1, ymax1 = region1
        xmin2, xmax2, ymin2, ymax2 = region2
        
        return (abs(xmin1 - xmin2) <= tolerance_x and 
                abs(xmax1 - xmax2) <= tolerance_x and
                abs(ymin1 - ymin2) <= tolerance_y and 
                abs(ymax1 - ymax2) <= tolerance_y)
    
    def unify_regions(self, raw_regions, tolerance_x=None, tolerance_y=None):
        """
        Unify continuous similar regions to prevent flickering.
        
        Args:
            raw_regions: Dict mapping frame_number -> list of coordinates
            tolerance_x: Horizontal pixel tolerance (defaults to config)
            tolerance_y: Vertical pixel tolerance (defaults to config)
            
        Returns:
            Dict with unified regions
        """
        if tolerance_x is None:
            tolerance_x = config.PIXEL_TOLERANCE_X
        if tolerance_y is None:
            tolerance_y = config.PIXEL_TOLERANCE_Y
        if len(raw_regions) == 0:
            return raw_regions
            
        keys = sorted(raw_regions.keys())
        unified_regions = {}
        
        # Initialize with first frame
        last_key = keys[0]
        unify_value_map = {last_key: raw_regions[last_key]}
        
        for key in keys[1:]:
            current_regions = raw_regions[key]
            new_unify_values = []
            
            for idx, region in enumerate(current_regions):
                # Get corresponding region from previous frame
                last_standard_region = (unify_value_map[last_key][idx] 
                                       if idx < len(unify_value_map[last_key]) 
                                       else None)
                
                # If current region is similar to previous frame's region, unify them
                if last_standard_region and self.are_similar(region, last_standard_region, 
                                                             tolerance_x, tolerance_y):
                    new_unify_values.append(last_standard_region)
                else:
                    new_unify_values.append(region)
            
            unify_value_map[key] = new_unify_values
            last_key = key
        
        # Build final unified result
        for key in keys:
            unified_regions[key] = unify_value_map[key]
            
        return unified_regions
    
    @staticmethod
    def get_coordinates(dt_box):
        """
        Convert detection boxes to coordinate tuples.
        
        Args:
            dt_box: Detection box results from PaddleOCR
            
        Returns:
            List of (xmin, xmax, ymin, ymax) tuples
        """
        coordinate_list = []
        if isinstance(dt_box, list):
            for box in dt_box:
                if isinstance(box, list) and len(box) >= 4:
                    (x1, y1) = int(box[0][0]), int(box[0][1])
                    (x2, y2) = int(box[1][0]), int(box[1][1])
                    (x3, y3) = int(box[2][0]), int(box[2][1])
                    (x4, y4) = int(box[3][0]), int(box[3][1])
                    
                    xmin = max(x1, x4)
                    xmax = min(x2, x3)
                    ymin = max(y1, y2)
                    ymax = min(y3, y4)
                    
                    coordinate_list.append((xmin, xmax, ymin, ymax))
        
        return coordinate_list
    
    def find_subtitle_frames(self):
        """
        Find all frames containing subtitles and their coordinates.
        NOTE: Expansion is done in create_mask, not here (matches backend behavior)
        
        Returns:
            Dict mapping frame_number -> list of (xmin, xmax, ymin, ymax) tuples
        """
        video_cap = cv2.VideoCapture(self.video_path)
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        subtitle_frames = {}
        current_frame = 0
        
        print('[Processing] Finding subtitles...')
        with tqdm(total=frame_count, unit='frame', desc='Subtitle Finding') as pbar:
            while video_cap.isOpened():
                ret, frame = video_cap.read()
                if not ret:
                    break
                
                current_frame += 1
                dt_boxes = self.detect_subtitle(frame)
                coordinates = self.get_coordinates(dt_boxes)
                
                if coordinates:
                    filtered_coords = []
                    for coord in coordinates:
                        xmin, xmax, ymin, ymax = coord
                        
                        # Filter by sub_area if specified
                        if self.sub_area is not None:
                            s_ymin, s_ymax, s_xmin, s_xmax = self.sub_area
                            if not (s_xmin <= xmin and xmax <= s_xmax and 
                                   s_ymin <= ymin and ymax <= s_ymax):
                                continue
                        
                        # Do NOT expand here - expansion happens in create_mask
                        filtered_coords.append(coord)
                    
                    if filtered_coords:
                        subtitle_frames[current_frame] = filtered_coords
                
                pbar.update(1)
        
        video_cap.release()
        
        # Unify regions across frames to prevent flickering (CRITICAL FIX)
        subtitle_frames = self.unify_regions(subtitle_frames)
        
        print('[Finished] Subtitle detection complete.')
        
        return subtitle_frames
