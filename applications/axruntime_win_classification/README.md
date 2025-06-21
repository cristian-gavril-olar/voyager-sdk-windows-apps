# Real-time Camera Classification

This application performs real-time image classification on your laptop camera feed using the Axelera AI runtime. It's designed for Windows 11 and provides live classification results overlaid on the camera stream.

## Features

- **Real-time Classification**: Classifies objects in live camera feed
- **Live Preview**: Shows camera feed with classification overlay
- **High Performance**: Optimized for smooth video with frame skipping
- **ImageNet Support**: Works with ImageNet-trained classification models
- **Easy Integration**: Based on the Axelera runtime example framework

## Visual Output

The application displays:
- **Live camera feed**: Your laptop's camera stream
- **Classification text**: Predicted class name in the top-right corner
- **Confidence score**: Prediction confidence as a percentage
- **Background overlay**: Semi-transparent background for better text readability

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
- ImageNet classification model (compiled for Axelera platform)
- Model should output 1000 classes (standard ImageNet)

## Installation

1. Ensure the Axelera environment is activated:
   ```bash
   source venv/bin/activate
   ```

2. Navigate to the application directory:
   ```bash
   cd applications/axruntime_win_classification
   ```

## Usage

### Basic Usage

```bash
python axruntime_win_classification.py /path/to/your/model.json
```

### Command Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `path` | str | Yes | - | Path to model.json file for ImageNet classification |
| `--camera-id` | int | No | 0 | Camera device ID |
| `--aipu-cores` | int | No | 4 | Number of AIPU cores to use |
| `--labels` | str | No | Auto-detected | Path to labels file |
| `-v, --verbose` | flag | No | - | Increase verbosity |

### Example Commands

**Basic classification:**
```bash
python axruntime_win_classification.py models/resnet50/model.json
```

**Using external camera:**
```bash
python axruntime_win_classification.py models/resnet50/model.json --camera-id 1
```

**With custom labels file:**
```bash
python axruntime_win_classification.py models/resnet50/model.json \
    --labels /path/to/custom_labels.txt
```

**Verbose output for debugging:**
```bash
python axruntime_win_classification.py models/resnet50/model.json --verbose
```

## Controls

- **Press 'q'**: Quit the application
- **Window**: The camera feed displays in a window titled "Camera Classification"

## Performance Features

### Optimization Strategies
- **Frame Skipping**: Processes every 3rd frame to maintain smooth video
- **Non-blocking Inference**: Uses threaded worker to prevent video lag
- **Queue Management**: Limited queue size prevents memory buildup
- **Efficient Preprocessing**: Optimized image preprocessing pipeline

### Real-time Display
- **Smooth Video**: 30 FPS camera feed with minimal lag
- **Instant Updates**: Classification results update as soon as inference completes
- **Visual Feedback**: Clear text overlay shows current predictions

## Model Requirements

### Input Specifications
- **Image Format**: RGB color images
- **Input Size**: Typically 224x224 or 640x640 (model-dependent)
- **Preprocessing**: Automatic letterbox resize, normalization, and quantization

### Output Specifications
- **Classes**: 1000 ImageNet classes
- **Format**: Single output tensor with class probabilities
- **Labels**: Uses ImageNet class labels by default

### Supported Models
The application works with any ImageNet classification model compiled for Axelera, including:
- ResNet variants (ResNet50, ResNet101, etc.)
- EfficientNet models
- MobileNet models
- Vision Transformer (ViT) models

## Labels File

The application automatically uses the default ImageNet labels file located at:
```
$AXELERA_FRAMEWORK/ax_datasets/labels/imagenet1000_clsidx_to_labels.txt
```

### Custom Labels
You can specify a custom labels file using the `--labels` argument:
```bash
python axruntime_win_classification.py model.json --labels my_labels.txt
```

The labels file should contain one label per line, with line numbers corresponding to class indices.

## Troubleshooting

### Camera Issues
- **Camera not found**: Try different `--camera-id` values (0, 1, 2, etc.)
- **Permission denied**: Ensure camera permissions are enabled
- **Black screen**: Check if another application is using the camera

### Model Issues
- **Model not found**: Verify the path to your `model.json` file
- **Unsupported model**: Ensure the model outputs 1000 classes for ImageNet
- **Labels mismatch**: Check that your labels file matches the model's classes

### Performance Issues
- **Slow classification**: The app processes every 3rd frame by default
- **High CPU usage**: Consider reducing frame rate or processing frequency
- **Memory issues**: Ensure sufficient system memory is available

## Technical Details

### Preprocessing Pipeline
1. **Resize**: Image resized to model input dimensions
2. **Color Space**: BGR to RGB conversion
3. **Normalization**: ImageNet mean/std normalization
4. **Quantization**: Conversion to int8 for Axelera runtime

### Classification Pipeline
1. **Frame Capture**: OpenCV captures frames at 30 FPS
2. **Frame Selection**: Every 3rd frame is processed
3. **Preprocessing**: Image prepared for model input
4. **Inference**: Model execution on Axelera AIPU
5. **Postprocessing**: Softmax and top-1 classification
6. **Display**: Results overlaid on live video

### Worker Thread Architecture
- **Main Thread**: Handles camera capture and display
- **Worker Thread**: Performs model inference
- **Queue System**: Non-blocking communication between threads

## Configuration

### Camera Settings
The application automatically configures the camera for optimal performance:
- Resolution: 640x480
- Frame Rate: 30 FPS
- Format: Default camera format

### Model Settings
- Batch Size: 1 (real-time processing)
- AIPU Cores: 4 (configurable)
- Precision: int8 quantized

## Related Files

- `axruntime_win_classification.py`: Main application script
- `../../examples/axruntime/axruntime_example.py`: Original example this is based on
- `../axruntime_win_ces2025/`: Related multi-model demo

## License

Copyright Axelera AI, 2025