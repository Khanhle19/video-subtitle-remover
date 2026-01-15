"""
Example usage of the lightweight OpenCV subtitle remover.

This script demonstrates how to use the module to remove subtitles from a video
using only OpenCV's Telea algorithm (no Deep Learning required).
"""

import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencv_subtitle_remover import SubtitleRemover


def main():
    # Get video path from command line or use default
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = input("Enter video file path: ").strip().strip('"')
    
    # Optional: Define subtitle area (ymin, ymax, xmin, xmax)
    # This restricts detection to a specific region of the frame
    # Example for bottom 20% of frame:
    # sub_area = (int(frame_height * 0.8), frame_height, 0, frame_width)
    sub_area = None  # None means detect in entire frame
    
    # Create remover instance
    print(f"Processing video: {video_path}")
    remover = SubtitleRemover(
        video_path=video_path,
        sub_area=sub_area,
        inpaint_radius=3  # Telea algorithm radius (3 is recommended)
    )
    
    # Remove subtitles
    output_path = remover.remove_subtitles()
    
    print(f"\n✓ Success! Output saved to: {output_path}")


if __name__ == "__main__":
    main()
