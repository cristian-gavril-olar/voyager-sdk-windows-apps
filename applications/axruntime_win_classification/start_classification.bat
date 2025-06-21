@echo off
REM Copyright Axelera AI, 2025
REM Windows batch file to start the axruntime_win_classification application
REM This file handles venv initialization and runs the application on the default camera

echo Starting Axelera Classification App...
echo.

REM Change to the voyager-sdk directory (two levels up from this script)
cd /d "%~dp0..\.."

REM Check if virtual environment exists
if not exist "venv-win\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at venv-win\Scripts\activate.bat
    echo Please ensure you have completed the Windows Getting Started Guide setup.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call "venv-win\Scripts\activate.bat"

REM Check if ResNet50 model exists
if not exist "build\resnet50-imagenet-onnx\resnet50-imagenet-onnx\1\model.json" (
    echo WARNING: ResNet50 model not found. Downloading now...
    echo Running: python download_prebuilt.py resnet50-imagenet-onnx
    python download_prebuilt.py resnet50-imagenet-onnx
    if %errorlevel% neq 0 (
        echo ERROR: Failed to download ResNet50 model
        echo Please run: python download_prebuilt.py resnet50-imagenet-onnx
        echo.
        pause
        exit /b 1
    )
)

REM Check if labels file exists
if not exist "examples\axruntime\imagenet-labels.txt" (
    echo ERROR: ImageNet labels file not found at examples\axruntime\imagenet-labels.txt
    echo Please ensure the voyager-sdk is properly installed.
    echo.
    pause
    exit /b 1
)

REM Start the classification application
echo Starting camera classification...
echo Press 'q' in the camera window to quit.
echo.

python.exe "applications\axruntime_win_classification\axruntime_win_classification.py" "build\resnet50-imagenet-onnx\resnet50-imagenet-onnx\1\model.json" --labels "examples\axruntime\imagenet-labels.txt" --camera-id 0

REM Check exit code
if %errorlevel% neq 0 (
    echo.
    echo Application exited with error code %errorlevel%
    echo.
    pause
    exit /b %errorlevel%
)

echo.
echo Classification application closed successfully.
pause