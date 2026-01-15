import warnings
from enum import Enum, unique
warnings.filterwarnings('ignore')
import os
import torch
import logging
import platform
import stat
from fsplit.filesplit import Filesplit
import onnxruntime as ort

# Project version
VERSION = "1.1.1"
# ×××××××××××××××××××× [DO NOT CHANGE] start ××××××××××××××××××××
logging.disable(logging.DEBUG)  # Disable DEBUG logging
logging.disable(logging.WARNING)  # Disable WARNING logging
try:
    import torch_directml
    device = torch_directml.device(torch_directml.default_device())
    USE_DML = True
except:
    USE_DML = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAMA_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'big-lama')
STTN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sttn', 'infer_model.pth')
VIDEO_INPAINT_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'video')
MODEL_VERSION = 'V4'
DET_MODEL_BASE = os.path.join(BASE_DIR, 'models')
DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det')

# Check if the complete model file exists in the path. If not, merge small files to generate the complete file
if 'big-lama.pt' not in (os.listdir(LAMA_MODEL_PATH)):
    fs = Filesplit()
    fs.merge(input_dir=LAMA_MODEL_PATH)

if 'inference.pdiparams' not in os.listdir(DET_MODEL_PATH):
    fs = Filesplit()
    fs.merge(input_dir=DET_MODEL_PATH)

if 'ProPainter.pth' not in os.listdir(VIDEO_INPAINT_MODEL_PATH):
    fs = Filesplit()
    fs.merge(input_dir=VIDEO_INPAINT_MODEL_PATH)

# Specify ffmpeg executable path
sys_str = platform.system()
if sys_str == "Windows":
    ffmpeg_bin = os.path.join('win_x64', 'ffmpeg.exe')
elif sys_str == "Linux":
    ffmpeg_bin = os.path.join('linux_x64', 'ffmpeg')
else:
    ffmpeg_bin = os.path.join('macos', 'ffmpeg')
FFMPEG_PATH = os.path.join(BASE_DIR, '', 'ffmpeg', ffmpeg_bin)

if 'ffmpeg.exe' not in os.listdir(os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64')):
    fs = Filesplit()
    fs.merge(input_dir=os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64'))
# Add executable permission to ffmpeg
os.chmod(FFMPEG_PATH, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Whether to use ONNX (DirectML/AMD/Intel)
ONNX_PROVIDERS = []
available_providers = ort.get_available_providers()
for provider in available_providers:
    if provider in [
        "CPUExecutionProvider"
    ]:
        continue
    if provider not in [
        "DmlExecutionProvider",         # DirectML, strictly for Windows GPU
        "ROCMExecutionProvider",        # AMD ROCm
        "MIGraphXExecutionProvider",    # AMD MIGraphX
        "VitisAIExecutionProvider",     # AMD VitisAI, for RyzenAI & Windows, performance seems similar to DirectML
        "OpenVINOExecutionProvider",    # Intel GPU
        "MetalExecutionProvider",       # Apple macOS
        "CoreMLExecutionProvider",      # Apple macOS
        "CUDAExecutionProvider",        # Nvidia GPU
    ]:
        continue
    ONNX_PROVIDERS.append(provider)
# ×××××××××××××××××××× [DO NOT CHANGE] end ××××××××××××××××××××


@unique
class InpaintMode(Enum):
    """
    Inpainting Algorithm Enum
    """
    STTN = 'sttn'
    LAMA = 'lama'
    PROPAINTER = 'propainter'


# ×××××××××××××××××××× [CAN CHANGE] start ××××××××××××××××××××
# Whether to use h264 encoding. If you need to share the generated video on Android phones, please enable this option
USE_H264 = True

# ×××××××××× General Settings start ××××××××××
"""
MODE Optional algorithm types
- InpaintMode.STTN algorithm: Better effect for real-person videos, fast speed, can skip subtitle detection
- InpaintMode.LAMA algorithm: Better effect for animation videos, average speed, cannot skip subtitle detection
- InpaintMode.PROPAINTER algorithm: Consumes a large amount of VRAM, slow speed, better effect for videos with very intense motion
"""
# [Set inpaint algorithm]
MODE = InpaintMode.STTN
# [Set pixel deviation]
# Used to determine if it is a non-subtitle area (generally assumes the subtitle text box length is greater than width. If height > width and exceeds the specified pixel size, it is considered a false detection)
THRESHOLD_HEIGHT_WIDTH_DIFFERENCE = 10
# Used to expand mask size, preventing the automatically detected text box from being too small, causing text edges to remain during inpainting
SUBTITLE_AREA_DEVIATION_PIXEL = 20
# Used to judge if two text boxes are the same line of subtitles. If height difference is within specified pixels, they are considered the same line
THRESHOLD_HEIGHT_DIFFERENCE = 20
# Used to judge if two subtitle text rectangles are similar. If diff in X and Y axes are within specified thresholds, they are considered the same text box
PIXEL_TOLERANCE_Y = 20  # Allowed vertical deviation pixels for detection box
PIXEL_TOLERANCE_X = 20  # Allowed horizontal deviation pixels for detection box
# ×××××××××× General Settings end ××××××××××

# ×××××××××× InpaintMode.STTN Algorithm Settings start ××××××××××
"""
1. STTN_SKIP_DETECTION
Meaning: Whether to skip detection
Effect: Setting to True skips subtitle detection, saving a lot of time, but may mistakenly injure non-subtitle video frames or cause missing subtitles to be removed

2. STTN_NEIGHBOR_STRIDE
Meaning: Neighboring frame stride. If missing area needs to be filled for the 50th frame, STTN_NEIGHBOR_STRIDE=5, then the algorithm will use the 45th frame, 40th frame, etc. as references.
Effect: Controls the density of reference frame selection. Larger stride means using fewer, more scattered reference frames; smaller stride means using more, more concentrated reference frames.

3. STTN_REFERENCE_LENGTH
Meaning: Number of reference frames. The STTN algorithm looks at several frames before and after each frame to be repaired to obtain context information.
Effect: Increasing this increases VRAM usage and improves processing effect, but slows down processing speed.

4. STTN_MAX_LOAD_NUM
Meaning: Maximum number of video frames loaded by STTN algorithm at a time
Effect: Larger setting means slower speed, but better effect
Note: Ensure STTN_MAX_LOAD_NUM is greater than STTN_NEIGHBOR_STRIDE and STTN_REFERENCE_LENGTH
"""
STTN_SKIP_DETECTION = True # Setting to True skips subtitle detection
STTN_NEIGHBOR_STRIDE = 5 # Reference frame stride
STTN_REFERENCE_LENGTH = 10 # Reference frame length (quantity)
STTN_MAX_LOAD_NUM = 90 # Set maximum frames processed simultaneously by STTN algorithm
if STTN_MAX_LOAD_NUM < STTN_REFERENCE_LENGTH * STTN_NEIGHBOR_STRIDE:
    STTN_MAX_LOAD_NUM = STTN_REFERENCE_LENGTH * STTN_NEIGHBOR_STRIDE
# ×××××××××× InpaintMode.STTN algorithm settings end ××××××××××

# ×××××××××× InpaintMode.PROPAINTER Algorithm Settings start ××××××××××
# [Set according to your GPU VRAM size] Maximum number of images processed simultaneously. Larger setting means better effect but requires higher VRAM
# 1280x720p video setting 80 requires 25G VRAM, setting 50 requires 19G VRAM
# 720x480p video setting 80 requires 8G VRAM, setting 50 requires 7G VRAM
PROPAINTER_MAX_LOAD_NUM = 70
# ×××××××××× InpaintMode.PROPAINTER Algorithm Settings end ××××××××××

# ×××××××××× InpaintMode.LAMA Algorithm Settings start ××××××××××
# Whether to enable super fast mode. Does not guarantee inpaint effect, only removes text in areas containing text
LAMA_SUPER_FAST = False
# ×××××××××× InpaintMode.LAMA Algorithm Settings end ××××××××××
# ×××××××××××××××××××× [CAN CHANGE] end ××××××××××××××××××××
