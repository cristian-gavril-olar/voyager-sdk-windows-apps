#!/usr/bin/env python
# Copyright Axelera AI, 2025
#
from __future__ import annotations

import argparse
import logging
from logging import getLogger
import os
from pathlib import Path
import queue
import threading
import time

from axelera.runtime import Context, TensorInfo
import cv2
import numpy as np

LOG = getLogger(__name__)

mean = [0.485, 0.456, 0.406]
stddev = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser(
    description="Real-time camera classification using Axelera runtime",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    add_help=False
)

parser.add_argument(
    "path",
    type=str,
    help="Path to model to test. This should be a model.json file for an imagenet classification model",
)
parser.add_argument("--aipu-cores", type=int, default=4, help="Number of AIPU cores to use")
parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID (default: 0)")
parser.add_argument(
    "--labels",
    type=Path,
    required=True,
    help="Path to text file containing labels. If you don't have a specific labels file, you can use [voyager-sdk-installation-path]/examples/axruntime/imagenet-labels.txt from your voyager-sdk installation.",
)
parser.add_argument(
    "-v",
    "--verbose",
    default=0,
    action="count",
    help="be more verbose; use repeatedly for more info",
)


def _preproc_frame(frame: np.ndarray, info: TensorInfo):
    batch, height, width, _ = info.unpadded_shape
    
    # Resize frame to model input size
    image = cv2.resize(frame, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = image / 255.0
    image = image - np.array(mean)
    image = image / np.array(stddev)
    quantized = np.round(image / info.scale + info.zero_point).clip(-128, 127).astype(np.int8)
    padded = np.pad(quantized, info.padding[1:], mode="constant", constant_values=info.zero_point)
    if batch > 1:
        padded = np.repeat(padded[np.newaxis, ...], batch, axis=0)
    return padded


def _postproc_frame(output: np.array, labels: list[str], info: TensorInfo):
    out = output[tuple(slice(b, -e if e else None) for b, e in info.padding)]
    out = out.squeeze()
    out = (out.astype(np.float32) - info.zero_point) * info.scale

    cls = np.argmax(out)
    label = labels[cls] if cls < len(labels) else "(no label)"
    score = out[cls]
    confidence = np.exp(score) / np.sum(np.exp(out)) * 100  # Softmax for percentage
    return cls, label, confidence


class CameraWorker(threading.Thread):
    def __init__(self, instance):
        self.instance = instance
        self.inqueue = queue.Queue(maxsize=2)  # Limit queue size to avoid lag
        self.outqueue = queue.Queue()
        self.running = True
        super().__init__()
        self.start()

    def run(self):
        while self.running:
            try:
                x = self.inqueue.get(timeout=0.1)
                if x is None:
                    break
                frame_id, *inputs_outputs = x
                try:
                    self.instance.run(*inputs_outputs)
                except Exception as e:
                    self.outqueue.put(e)
                    break
                else:
                    self.outqueue.put((frame_id, inputs_outputs[1]))
            except queue.Empty:
                continue

    def push(self, frame_id, inputs, outputs):
        try:
            self.inqueue.put([frame_id, inputs, outputs], block=False)
            return True
        except queue.Full:
            return False  # Skip frame if queue is full

    def pop(self):
        try:
            x = self.outqueue.get(block=False)
            if isinstance(x, Exception):
                raise x
            return x
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.inqueue.put(None)


def run_camera_classification(
    model_path: Path,
    aipu_cores: int,
    camera_id: int,
    labels: list[str],
):
    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_id}")
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    with Context() as ctx:
        model = ctx.load_model(model_path)

        input_infos, output_infos = model.inputs(), model.outputs()
        output_shapes = [i.shape for i in output_infos]
        assert len(output_shapes) == 1, "Only one output shape supported"
        assert output_shapes[0][1:-1] == (1, 1), "Only 1000 classes supported"
        batch_size = input_infos[0].shape[0]

        num_instances = min(aipu_cores // batch_size, 1)  # Use single instance for real-time
        if num_instances == 0:
            num_instances = 1
            batch_size = 1

        # Create connection to AIPU
        connection = ctx.device_connect(None, batch_size)
        LOG.info(f"Creating model instance with batch size of {batch_size}")
        instance = connection.load_model_instance(
            model,
            num_sub_devices=batch_size,
            aipu_cores=batch_size,
        )

        # Create input and output buffers
        inputs = [np.zeros(t.shape, np.int8) for t in input_infos]
        outputs = [np.zeros(t.shape, np.int8) for t in output_infos]
        worker = CameraWorker(instance)

        current_classification = "Initializing..."
        current_confidence = 0.0
        frame_count = 0
        
        try:
            print("Starting camera classification. Press 'q' to quit.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Display current frame with classification overlay
                display_frame = frame.copy()
                
                # Add classification text in top right corner
                text = f"{current_classification}"
                confidence_text = f"Confidence: {current_confidence:.1f}%"
                
                # Get text size for positioning
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                (conf_width, conf_height), conf_baseline = cv2.getTextSize(confidence_text, font, font_scale, thickness)
                
                # Position in top right corner
                text_x = display_frame.shape[1] - max(text_width, conf_width) - 10
                text_y = 30
                conf_y = text_y + text_height + 10
                
                # Add background rectangles for better readability
                cv2.rectangle(display_frame, 
                            (text_x - 5, text_y - text_height - 5), 
                            (text_x + max(text_width, conf_width) + 5, conf_y + conf_height + 5), 
                            (0, 0, 0), -1)
                
                # Add text
                cv2.putText(display_frame, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
                cv2.putText(display_frame, confidence_text, (text_x, conf_y), font, font_scale, (0, 255, 0), thickness)
                
                # Show the frame
                cv2.imshow('Camera Classification', display_frame)
                
                # Process every 3rd frame to reduce computational load
                if frame_count % 3 == 0:
                    # Preprocess frame for model inference
                    processed_input = _preproc_frame(frame, input_infos[0])
                    inputs[0][:] = processed_input
                    
                    # Push to worker (non-blocking)
                    worker.push(frame_count, inputs, outputs)
                
                # Check for results
                result = worker.pop()
                if result is not None:
                    frame_id, outs = result
                    cls, label, confidence = _postproc_frame(outs[0], labels, output_infos[0])
                    current_classification = label
                    current_confidence = confidence
                    LOG.debug(f"Frame {frame_id}: {label} ({confidence:.1f}%)")

                frame_count += 1
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            worker.stop()
            worker.join()
            cap.release()
            cv2.destroyAllWindows()


def main(args: argparse.Namespace):
    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    desired = levels.get(args.verbose, logging.DEBUG)
    logging.basicConfig(level=desired)

    model_path = Path(args.path)
    labels = args.labels.read_text().splitlines()
    if model_path.is_dir():
        model_path /= "model.json"
    
    try:
        run_camera_classification(
            model_path,
            args.aipu_cores,
            args.camera_id,
            labels,
        )
    except Exception as e:
        if args.verbose:
            raise
        print(f'FAIL: {e}')
        return 1
    else:
        return 0


def entrypoint_main():
    try:
        args = parser.parse_args()
    except SystemExit:
        parser.print_help()
        return 1
    
    try:
        return main(args)
    except RuntimeError as e:
        if args.verbose:
            raise
        print(f'ERROR: {e}')
        return 1


if __name__ == '__main__':
    exit(entrypoint_main())