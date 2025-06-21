# Axelera Detection Runtime Example

This application demonstrates how to run object detection models using the Axelera Runtime on AIPU hardware. It performs object detection on images and outputs bounding box coordinates and class names to the console.

## Features

- Supports COCO-trained object detection models
- Console output with bounding box coordinates in pixel space
- Built-in Non-Maximum Suppression (NMS) for clean detection results
- Multi-threaded inference for optimal AIPU utilization
- Configurable confidence and NMS thresholds

## Requirements

- Axelera environment activated (`source venv/bin/activate`)
- OpenCV Python package (`cv2`)
- COCO-trained detection model (e.g., YOLOv8, SSD-MobileNet)

## Usage

```bash
python axruntime_detection_example.py MODEL_PATH IMAGE_PATHS [OPTIONS]
```

### Arguments

- `MODEL_PATH`: Path to the detection model (model.json file or directory containing it)
- `IMAGE_PATHS`: One or more paths to images or directories containing images

### Options

- `--labels PATH`: Path to labels file (default: `$AXELERA_FRAMEWORK/ax_datasets/labels/coco.names`)
- `--conf-threshold FLOAT`: Confidence threshold for detections (default: 0.25)
- `--nms-threshold FLOAT`: NMS IoU threshold (default: 0.45)
- `--max-detections INT`: Maximum number of detections to keep after NMS (default: 100)
- `--aipu-cores INT`: Number of AIPU cores to use (default: 4)
- `-v, --verbose`: Increase verbosity (use multiple times for more detail)

## Example Commands

### Basic detection on a single image:
```bash
python axruntime_detection_example.py /path/to/yolov8n-coco model.json image.jpg
```

### Detection on multiple images with custom thresholds:
```bash
python axruntime_detection_example.py /path/to/model images/ --conf-threshold 0.5 --nms-threshold 0.4
```

### Using custom labels file:
```bash
python axruntime_detection_example.py /path/to/model image.jpg --labels /path/to/custom_labels.txt
```

## Labels File

The application uses COCO labels by default from:
```
$AXELERA_FRAMEWORK/ax_datasets/labels/coco.names
```

This file contains 80 object class names including:
- person
- bicycle  
- car
- motorcycle
- airplane
- bus
- train
- truck
- boat
- traffic light
- ... and 70 more classes

For custom models trained on different datasets, provide a text file with one class name per line using the `--labels` parameter.

## Output Format

The application outputs detection results to the console in the following format:

```
image.jpg: Found 3 detections:
  Detection 1: person (confidence: 0.892) bbox: [145.2, 67.8, 234.1, 456.3] (pixel coordinates)
  Detection 2: car (confidence: 0.758) bbox: [312.4, 189.1, 187.6, 298.7] (pixel coordinates)  
  Detection 3: bicycle (confidence: 0.634) bbox: [89.3, 234.5, 156.8, 201.2] (pixel coordinates)
```

Where bbox format is `[x, y, width, height]` in pixel coordinates relative to the input image dimensions.

## Model Compatibility

This application is designed for COCO-trained object detection models including:

- **YOLOv8 series**: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
- **YOLOv5 series**: Various YOLOv5 models
- **SSD MobileNet**: SSD-MobileNetV1, SSD-MobileNetV2
- **Other detection models** with YOLO-style output format

The models should output detection results in the format:
- `[x_center, y_center, width, height, objectness, class_scores...]`
- With 80 classes for COCO dataset (total 85 values per detection)

## Technical Details

### Preprocessing
- Images are resized to model input dimensions
- RGB color format conversion
- Normalization using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- INT8 quantization according to model requirements

### Postprocessing
- Automatic sigmoid activation for raw logit outputs
- Coordinate conversion from model space to pixel space
- Per-class Non-Maximum Suppression for optimal results
- Confidence-based filtering
- Bounding box validation and clipping

### Performance
- Multi-threaded inference using multiple AIPU cores
- Automatic load balancing across available hardware
- Batch processing support where applicable

## Troubleshooting

### Common Issues

1. **"Could not load image"**: Ensure image path is correct and file format is supported (jpg, png, bmp, tiff, webp)

2. **"No detections found"**: Try lowering the confidence threshold with `--conf-threshold 0.1`

3. **"Feature map output detected"**: Model may use anchor-based detection format which requires different decoding

4. **Environment errors**: Ensure Axelera environment is activated with `source venv/bin/activate`

### Debug Mode

Use `-v` or `-vv` for verbose output to see:
- Model loading details
- Preprocessing information  
- Raw detection scores and coordinates
- NMS filtering results