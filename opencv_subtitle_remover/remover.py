"""
OpenCV-based subtitle removal using Telea inpainting algorithm.

This module provides a lightweight alternative to Deep Learning models.
"""

import os
import cv2
import tempfile
import subprocess
from tqdm import tqdm
from pathlib import Path

from . import config
from .detector import SubtitleDetector
from .utils import create_mask, is_video_or_image


def get_ffmpeg_path():
    """Get the path to ffmpeg executable."""
    # Try to use bundled ffmpeg first
    current_dir = Path(__file__).parent.parent
    
    # Check for complete ffmpeg.exe
    ffmpeg_exe = current_dir / 'backend' / 'ffmpeg' / 'win_x64' / 'ffmpeg.exe'
    if ffmpeg_exe.exists():
        return str(ffmpeg_exe)
    
    # Fallback to system ffmpeg
    # If ffmpeg is not in PATH, user needs to install it
    return 'ffmpeg'


class SubtitleRemover:
    """
    Remove subtitles from videos using OpenCV's Telea inpainting algorithm.
    """
    
    def __init__(self, video_path, sub_area=None, inpaint_radius=None):
        """
        Initialize subtitle remover.
        
        Args:
            video_path: Path to input video file
            sub_area: Optional tuple (ymin, ymax, xmin, xmax) to restrict detection area
            inpaint_radius: Radius for Telea inpainting algorithm (default: from config)
        """
        if inpaint_radius is None:
            inpaint_radius = config.INPAINT_RADIUS
        if not is_video_or_image(video_path):
            raise ValueError(f"Invalid video path: {video_path}")
        
        self.video_path = video_path
        self.sub_area = sub_area
        self.inpaint_radius = inpaint_radius
        
        # Video properties
        self.video_cap = cv2.VideoCapture(video_path)
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output setup
        self.output_path = self._get_output_path()
        self.temp_file = tempfile.NamedTemporaryFile(
            suffix='.mp4', 
            delete=False,
            dir=os.path.dirname(video_path)
        )
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.temp_file.name,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        # Initialize detector
        self.detector = SubtitleDetector(video_path, sub_area)
    
    def _get_output_path(self):
        """Generate output file path."""
        base_dir = os.path.dirname(self.video_path)
        filename = os.path.basename(self.video_path)
        name, ext = os.path.splitext(filename)
        return os.path.join(base_dir, f"{name}_no_sub{ext}")
    
    def remove_subtitles(self):
        """
        Remove subtitles from the video using OpenCV Telea algorithm.
        
        Returns:
            str: Path to output video file
        """
        print('[Step 1/2] Detecting subtitles...')
        subtitle_frames = self.detector.find_subtitle_frames()
        
        print(f'[Info] Found subtitles in {len(subtitle_frames)} frames')
        
        print('[Step 2/2] Removing subtitles...')
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
        
        frame_idx = 0
        with tqdm(total=self.frame_count, unit='frame', desc='Subtitle Removing') as pbar:
            while True:
                ret, frame = self.video_cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                
                # Apply inpainting if this frame has subtitles
                if frame_idx in subtitle_frames:
                    mask = create_mask(
                        (self.frame_height, self.frame_width),
                        subtitle_frames[frame_idx]
                    )
                    frame = cv2.inpaint(
                        frame, 
                        mask, 
                        self.inpaint_radius, 
                        cv2.INPAINT_TELEA
                    )
                
                self.video_writer.write(frame)
                pbar.update(1)
        
        # Cleanup
        self.video_cap.release()
        self.video_writer.release()
        
        # Merge audio back
        print('[Step 3/3] Merging audio...')
        self._merge_audio()
        
        print(f'[Finished] Output saved to: {self.output_path}')
        return self.output_path
    
    def _merge_audio(self):
        """Merge original audio into the processed video."""
        ffmpeg_cmd = get_ffmpeg_path()
        
        # Check if ffmpeg is available
        try:
            subprocess.run([ffmpeg_cmd, '-version'], 
                         capture_output=True, check=True, timeout=5)
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            print('[Warning] ffmpeg not found. Saving video without audio.')
            import shutil
            shutil.copy2(self.temp_file.name, self.output_path)
            return
        
        # Extract audio from original video
        audio_temp = tempfile.NamedTemporaryFile(suffix='.aac', delete=False)
        
        try:
            # Extract audio
            extract_cmd = [
                ffmpeg_cmd, '-y', '-i', self.video_path,
                '-acodec', 'copy', '-vn',
                '-loglevel', 'error',
                audio_temp.name
            ]
            subprocess.run(extract_cmd, check=True, shell=False)
            
            # Merge audio with processed video
            merge_cmd = [
                ffmpeg_cmd, '-y',
                '-i', self.temp_file.name,
                '-i', audio_temp.name,
                '-vcodec', 'copy',
                '-acodec', 'copy',
                '-loglevel', 'error',
                self.output_path
            ]
            subprocess.run(merge_cmd, check=True, shell=False)
            
        except subprocess.CalledProcessError as e:
            print(f'[Warning] Failed to merge audio: {e}. Copying video without audio.')
            import shutil
            shutil.copy2(self.temp_file.name, self.output_path)
        
        finally:
            # Cleanup temp files
            try:
                os.remove(audio_temp.name)
                os.remove(self.temp_file.name)
            except:
                pass
