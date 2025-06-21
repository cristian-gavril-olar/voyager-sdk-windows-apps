# CES2025 Demo - Multi-Model Computer Vision Pipeline

This application demonstrates a real-time computer vision pipeline using three YOLOv8 models running simultaneously on the Axelera AI platform. It captures video from your laptop camera and performs pose detection, object detection, and segmentation in parallel.

## Features

- **Real-time Pose Detection**: Detects human poses with 17 COCO keypoints
- **Object Detection**: Identifies and tracks human subjects
- **Instance Segmentation**: Ready for segmentation tasks (framework included)
- **Live Visualization**: Displays results with color-coded overlays
- **Multi-threaded Processing**: Each model runs in its own thread for optimal performance

## Visual Output

The application overlays the following on the live camera feed:

- **Yellow lines**: Skeleton connections showing human pose structure
- **Red dots**: Key body parts (eyes, ears, wrists, ankles)
- **Thick red boxes**: Bounding boxes around detected humans
- **Info text**: Frame count and detection statistics

## Requirements

### Hardware
- Axelera AI hardware platform
- USB camera or built-in laptop camera
- Windows 11 (recommended)

### Software
- Python 3.8+
- Axelera SDK environment activated
- OpenCV Python (`cv2`)
- NumPy

### Models
You need three compiled YOLOv8 models:
1. **YOLOv8 Pose Model** (`yolov8lpose`) - for pose detection
2. **YOLOv8 Detection Model** (`yolov8s`) - for object detection  
3. **YOLOv8 Segmentation Model** (`yolov8sseg`) - for segmentation

## Installation

1. Ensure the Axelera environment is activated:
   ```bash
   source venv/bin/activate
   ```

2. Navigate to the application directory:
   ```bash
   cd applications/axruntime_win_ces2025
   ```

## Usage

### Basic Usage

```bash
python axruntime_win_ces2025.py \
    --pose-model /path/to/yolov8lpose/model.json \
    --detect-model /path/to/yolov8s/model.json \
    --seg-model /path/to/yolov8sseg/model.json
```

### Command Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--pose-model` | str | Yes | - | Path to YOLOv8 pose model (model.json) |
| `--detect-model` | str | Yes | - | Path to YOLOv8 detection model (model.json) |
| `--seg-model` | str | Yes | - | Path to YOLOv8 segmentation model (model.json) |
| `--camera-id` | int | No | 0 | Camera device ID |
| `--aipu-cores` | int | No | 4 | Number of AIPU cores to use |
| `-v, --verbose` | flag | No | - | Increase verbosity (use multiple times) |

### Example Commands

**Using default camera:**
```bash
python axruntime_win_ces2025.py \
    --pose-model models/yolov8lpose/model.json \
    --detect-model models/yolov8s/model.json \
    --seg-model models/yolov8sseg/model.json
```

**Using external camera:**
```bash
python axruntime_win_ces2025.py \
    --pose-model models/yolov8lpose/model.json \
    --detect-model models/yolov8s/model.json \
    --seg-model models/yolov8sseg/model.json \
    --camera-id 1
```

**Running directly from project root:**

```cmd
python applications/axruntime_win_ces2025/axruntime_win_ces2025.py --pose-model  .\build\yolov8lpose-coco-onnx\yolov8lpose-coco-onnx\1\model.json --detect-model  .\build\yolov8l-coco-onnx\yolov8l-coco-onnx\1\model.json --seg-model  .\build\yolov8sseg-coco-onnx\yolov8sseg-coco-onnx\1\model.json --camera-id 0
```

**With verbose output:**
```bash
python axruntime_win_ces2025.py \
    --pose-model models/yolov8lpose/model.json \
    --detect-model models/yolov8s/model.json \
    --seg-model models/yolov8sseg/model.json \
    --verbose
```

## Controls

- **Press 'q'**: Quit the application
- **Window**: The camera feed displays in a window titled "CES2025 Demo - Pose + Detection"

## Performance Optimization

The application includes several optimizations:

- **Frame Skipping**: Processes every 3rd frame to maintain smooth video
- **Multi-threading**: Each model runs in a separate thread
- **Queue Management**: Limited queue sizes prevent memory buildup
- **Letterbox Preprocessing**: Maintains aspect ratios while resizing

## Model Configuration

The application expects models with these specifications:

### Pose Model (YOLOv8 Pose)
- Input: RGB image (640x640)
- Output: Bounding boxes + 17 COCO keypoints per person
- AIPU cores: 2 (recommended)

### Detection Model (YOLOv8s)  
- Input: RGB image (640x640)
- Output: Bounding boxes for 80 classes (filters for person class)
- AIPU cores: 1 (recommended)

### Segmentation Model (YOLOv8s Segmentation)
- Input: RGB image (640x640) 
- Output: Instance segmentation masks
- AIPU cores: 1 (recommended)

## Troubleshooting

### Camera Issues
- **Camera not found**: Try different `--camera-id` values (0, 1, 2, etc.)
- **Permission denied**: Ensure camera permissions are enabled for the terminal/application

### Model Issues
- **Model not found**: Verify model paths point to valid `model.json` files
- **AIPU cores**: Adjust `--aipu-cores` based on your hardware configuration

### Performance Issues
- **Slow performance**: Reduce frame processing frequency in the code
- **Memory issues**: Ensure sufficient system memory is available

## Technical Details

### COCO Keypoints (17 points)
0. nose, 1. left_eye, 2. right_eye, 3. left_ear, 4. right_ear,
5. left_shoulder, 6. right_shoulder, 7. left_elbow, 8. right_elbow,
9. left_wrist, 10. right_wrist, 11. left_hip, 12. right_hip,
13. left_knee, 14. right_knee, 15. left_ankle, 16. right_ankle

### Pipeline Architecture
1. **Camera Capture**: OpenCV captures frames from camera
2. **Preprocessing**: Letterbox resize and normalization for each model
3. **Inference**: Parallel execution on separate AIPU cores
4. **Postprocessing**: NMS, coordinate scaling, and result parsing
5. **Visualization**: Overlay generation and display

## Related Files

- `axruntime_win_ces2025.py`: Main application script
- `../axruntime_win_classification/`: Related classification demo

## License

Copyright Axelera AI, 2025