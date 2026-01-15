# Lightweight OpenCV Subtitle Remover

A standalone, lightweight module for removing subtitles from videos using only OpenCV's Telea inpainting algorithm.

## Features

- ✅ **No Deep Learning dependencies** (PyTorch, ONNX, etc.)
- ✅ **Minimal requirements** (OpenCV, PaddleOCR for detection)
- ✅ **Fast processing** on CPU
- ✅ **Portable** - easy to integrate into other projects
- ✅ **Self-contained** module structure

## Installation

```bash
cd opencv_subtitle_remover
pip install -r requirements.txt
```

## Quick Start

```python
from opencv_subtitle_remover import SubtitleRemover

# Create remover instance
remover = SubtitleRemover(
    video_path="input_video.mp4",
    sub_area=None,  # Optional: (ymin, ymax, xmin, xmax)
    inpaint_radius=3
)

# Remove subtitles
output_path = remover.remove_subtitles()
print(f"Output: {output_path}")
```

Or use the example script:

```bash
python example.py path/to/video.mp4
```

## How It Works

1. **Detection**: Uses PaddleOCR to detect text boxes in each frame
2. **Inpainting**: Applies OpenCV's Telea algorithm (`cv2.INPAINT_TELEA`) to remove detected text
3. **Audio Merging**: Merges original audio back into the processed video

## Algorithm

This module uses **Alexandru Telea's Fast Marching Method** for inpainting:
- Propagates pixel colors from the boundary inward
- Very fast on CPU
- Good for simple backgrounds
- Trade-off: Lower quality than Deep Learning models for complex scenes

## Performance

- **Speed**: ~50-100x faster than Deep Learning models on CPU
- **Memory**: <100MB RAM (vs 2-8GB for DL models)
- **Quality**: Good for static backgrounds, may show artifacts on complex scenes

## Module Structure

```
opencv_subtitle_remover/
├── __init__.py          # Module exports
├── detector.py          # Subtitle detection (PaddleOCR)
├── remover.py           # OpenCV Telea inpainting
├── utils.py             # Helper functions
├── requirements.txt     # Minimal dependencies
├── example.py           # Usage example
└── README.md            # This file
```

## Requirements

- Python 3.7+
- opencv-python
- paddlepaddle
- paddleocr
- numpy
- tqdm

**Note**: FFmpeg must be installed on your system for audio merging.

## Limitations

- Quality is lower than Deep Learning models (STTN, ProPainter)
- Best for videos with simple, static backgrounds
- May leave visible artifacts on complex scenes
- Requires PaddleOCR models for text detection

## Integration

This module is designed to be portable. To use in another project:

1. Copy the `opencv_subtitle_remover/` folder to your project
2. Install dependencies: `pip install -r opencv_subtitle_remover/requirements.txt`
3. Import and use:

```python
from opencv_subtitle_remover import SubtitleRemover
```

## License

Same as parent project (video-subtitle-remover).
