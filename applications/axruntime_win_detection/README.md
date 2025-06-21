# Object Detection Application for Windows

This application provides object detection capabilities using the Axelera runtime, specifically designed for Windows environments. It processes images through compiled detection models and displays results with bounding boxes and confidence scores in the console.

## Features

- **Multi-threaded Processing**: Utilizes multiple AIPU cores for efficient inference
- **Advanced NMS**: Per-class Non-Maximum Suppression for cleaner results
- **Flexible Input**: Supports single images, multiple images, or entire directories
- **Configurable Thresholds**: Adjustable confidence and NMS thresholds
- **Visual Output**: Draws bounding boxes with labels on images and saves to output directory
- **Windows PowerShell Compatible**: Optimized for Windows PowerShell usage

## Requirements

- Windows environment with PowerShell
- Axelera SDK properly installed and configured
- Compiled object detection model (YOLO, SSD, etc.)
- Python environment with required dependencies

## Installation

Ensure you have activated the Axelera environment:

```powershell
# Activate the virtual environment (adjust path as needed)
.\venv\Scripts\Activate.ps1
```

## Basic Usage

### Command Syntax

```powershell
python.exe .\applications\axruntime_win_detection\axruntime_detection_example.py MODEL_PATH IMAGE_PATHS [OPTIONS]
```

### Required Arguments

- `MODEL_PATH`: Path to the compiled detection model (model.json file or directory containing it)
- `IMAGE_PATHS`: One or more paths to images or directories containing images

### Optional Arguments

- `--labels LABELS_FILE`: Path to text file containing class labels
- `--aipu-cores CORES`: Number of AIPU cores to use (default: 4)
- `--conf-threshold THRESHOLD`: Confidence threshold for detections (default: 0.25)
- `--nms-threshold THRESHOLD`: NMS IoU threshold (default: 0.45)
- `--max-detections COUNT`: Maximum number of detections to keep (default: 100)
- `--output-dir DIR`: Output directory for images with drawn bounding boxes (default: images_out)
- `--save-images`: Save images with drawn bounding boxes to output directory
- `-v, --verbose`: Increase verbosity (use multiple times for more detail)

## Examples

### 1. Basic Detection on Single Image

```powershell
python.exe .\applications\axruntime_win_detection\axruntime_detection_example.py .\build\yolov8l-coco-onnx\yolov8l-coco-onnx\1\model.json .\examples\axruntime\images\car.jpg --labels .\ax_datasets\labels\coco.names
```

### 2. Process Multiple Images with Visual Output

```powershell
python.exe .\applications\axruntime_win_detection\axruntime_detection_example.py .\build\yolov8l-coco-onnx\yolov8l-coco-onnx\1\model.json .\examples\axruntime\images --labels .\ax_datasets\labels\coco.names --save-images --output-dir results
```

### 3. Process with Custom Thresholds and Save Images

```powershell
python.exe .\applications\axruntime_win_detection\axruntime_detection_example.py .\build\yolov8l-coco-onnx\yolov8l-coco-onnx\1\model.json .\examples\axruntime\images --labels .\ax_datasets\labels\coco.names --conf-threshold 0.5 --nms-threshold 0.4 --max-detections 50 --save-images
```

### 4. High-Confidence Detections Only

```powershell
python.exe .\applications\axruntime_win_detection\axruntime_detection_example.py .\build\yolov8l-coco-onnx\yolov8l-coco-onnx\1\model.json .\examples\axruntime\images --labels .\ax_datasets\labels\coco.names --conf-threshold 0.7 --max-detections 20 --save-images -v
```

### 5. Process Entire Directory with Verbose Output

```powershell
python.exe .\applications\axruntime_win_detection\axruntime_detection_example.py .\build\yolov8l-coco-onnx\yolov8l-coco-onnx\1\model.json "C:\Users\YourName\Pictures\TestImages" --labels .\ax_datasets\labels\coco.names --save-images --output-dir "C:\Results\Detection" -vv
```

## Label Files

### Supported Formats

Label files should contain one class name per line:

```
person
bicycle
car
motorcycle
airplane
bus
train
truck
...
```

### Available Label Files

The SDK includes several pre-defined label files:

- `.\ax_datasets\labels\coco.names` - COCO dataset (80 classes)
- `.\ax_datasets\labels\coco90.names` - COCO dataset (90 classes)
- `.\ax_datasets\labels\pascalvoc.names` - Pascal VOC dataset
- `.\ax_datasets\labels\imagenet1000_clsidx_to_labels.txt` - ImageNet classes

### Label File Fallback

If no label file is specified or found, the application will:
1. Try to use COCO labels automatically
2. Generate generic class names (`class_0`, `class_1`, etc.) as fallback

## Output Format

The application displays detection results in the console:

```
.\examples\axruntime\images\car.jpg: Found 3 detections:
  Detection 1: car (confidence: 0.847) bbox: [123.4, 56.7, 89.1, 234.5]
  Detection 2: person (confidence: 0.731) bbox: [456.7, 123.4, 234.5, 178.9]
  Detection 3: traffic light (confidence: 0.623) bbox: [789.0, 234.5, 123.4, 156.7]
```

### Output Components

- **Image Path**: Full path to the processed image
- **Detection Count**: Total number of objects found after NMS
- **Class Name**: Object class from the label file
- **Confidence**: Detection confidence score (0.0 to 1.0)
- **Bounding Box**: `[x, y, width, height]` in pixel coordinates

## Visual Output

When using the `--save-images` flag, the application creates an output directory (default: `images_out`) containing:

### Generated Files

- **`detected_[original_filename]`**: Images with detections, showing:
  - Colored bounding boxes around detected objects
  - Class labels with confidence scores
  - Consistent colors for each object class
  
- **`no_detections_[original_filename]`**: Original images copied when no objects are detected

### Visual Features

- **Color-coded boxes**: Each object class gets a consistent, unique color
- **Labels with confidence**: Each detection shows "ClassName: 0.XX" format
- **Clean presentation**: White text on colored background for readability
- **Preserved quality**: Output images maintain original resolution

### Example Output Structure

```
images_out/
├── detected_car.jpg          # Car image with bounding boxes
├── detected_elephant.jpg     # Elephant image with bounding boxes
├── detected_hedgehog.jpg     # Hedgehog image with bounding boxes
└── no_detections_blank.jpg   # Image with no detections found
```

## Parameter Tuning Guide

### Confidence Threshold (`--conf-threshold`)

- **0.1-0.3**: Very permissive, shows weak detections
- **0.25**: Default, balanced detection
- **0.5-0.7**: Conservative, only high-confidence detections
- **0.8+**: Very strict, only strongest detections

### NMS Threshold (`--nms-threshold`)

- **0.3-0.4**: Aggressive NMS, removes more overlapping boxes
- **0.45**: Default, balanced overlap removal
- **0.5-0.6**: Permissive NMS, allows some overlap
- **0.7+**: Minimal NMS, keeps most boxes

### Max Detections (`--max-detections`)

- **10-20**: Minimal output, top detections only
- **50-100**: Moderate output (default: 100)
- **200+**: Comprehensive output, all valid detections

## Troubleshooting

### Common Issues

1. **"ModelInstance.run() takes 3 positional arguments but 4 were given"**
   - This has been fixed in the current version
   - Ensure you're using the latest version of the application

2. **"Failed to load model"**
   - Check that the model path is correct
   - Ensure the model is properly compiled for AIPU
   - Verify the model.json file exists

3. **"No detections found"**
   - Try lowering `--conf-threshold` (e.g., 0.1)
   - Check if the model expects different input preprocessing
   - Verify the model is compatible with your images

4. **"Could not load image"**
   - Ensure image files are in supported formats (JPG, PNG, BMP)
   - Check file permissions and paths
   - Use absolute paths if relative paths cause issues

### Debug Mode

Use verbose flags for detailed information:

```powershell
# Basic verbose mode
python.exe .\applications\axruntime_win_detection\axruntime_detection_example.py MODEL_PATH IMAGES --labels LABELS -v

# Maximum verbosity
python.exe .\applications\axruntime_win_detection\axruntime_detection_example.py MODEL_PATH IMAGES --labels LABELS -vv
```

This will show:
- Model loading details
- Input/output tensor information
- Processing pipeline status
- Detailed error messages

## Performance Tips

### Optimal Settings

For **speed** (faster processing):
```powershell
--conf-threshold 0.5 --max-detections 20 --aipu-cores 4
```

For **accuracy** (comprehensive detection):
```powershell
--conf-threshold 0.25 --nms-threshold 0.45 --max-detections 100
```

For **clean output** (minimal overlaps):
```powershell
--conf-threshold 0.4 --nms-threshold 0.3 --max-detections 50
```

### Hardware Utilization

- Use `--aipu-cores 4` for maximum performance (default)
- Reduce cores if processing small batches or single images
- Monitor system resources during processing

## Integration

This application can be used as:

1. **Standalone tool** for batch image processing
2. **Development utility** for model validation
3. **Integration component** in larger Windows workflows
4. **Testing framework** for detection model evaluation

For integration with the main Axelera pipeline, refer to the main `inference.py` documentation and the `examples/axruntime/README_detection.md` file.

## Support

For issues and questions:
- Check the main SDK documentation
- Review the `examples/axruntime/README_detection.md` for additional usage patterns
- Ensure all dependencies are properly installed
- Verify the Axelera environment is correctly activated