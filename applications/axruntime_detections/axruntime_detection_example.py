#!/usr/bin/env python
# Copyright Axelera AI, 2025
#
from __future__ import annotations

import argparse
import collections
import logging
from logging import getLogger
import os
from pathlib import Path
import queue
import threading

from axelera.runtime import Context, TensorInfo
import cv2  # noqa
import numpy as np

LOG = getLogger(__name__)

# Preprocessing values for COCO-trained detection models
mean = [0.485, 0.456, 0.406]
stddev = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)

parser.add_argument(
    "path",
    type=str,
    help="Path to model to test. This should be a model.json file for an object detection model",
)
parser.add_argument(
    "input_paths", type=Path, nargs='+', help="Path(s) to images or directories containing images"
)
parser.add_argument("--aipu-cores", type=int, default=4, help="Number of AIPU cores to use")
_DEFAULT_LABELS = os.path.expandvars(
    "$AXELERA_FRAMEWORK/ax_datasets/labels/coco.names"
)
_DEFAULT_LABELS = os.path.relpath(_DEFAULT_LABELS)
parser.add_argument(
    "--labels",
    type=Path,
    default=_DEFAULT_LABELS,
    help="Path to text file containing labels (default: %(default)s)",
)
parser.add_argument(
    "--conf-threshold",
    type=float,
    default=0.25,
    help="Confidence threshold for detections (default: %(default)s)",
)
parser.add_argument(
    "--nms-threshold",
    type=float,
    default=0.45,
    help="NMS IoU threshold (default: %(default)s)",
)
parser.add_argument(
    "--max-detections",
    type=int,
    default=100,
    help="Maximum number of detections to keep after NMS (default: %(default)s)",
)
parser.add_argument(
    "-v",
    "--verbose",
    default=0,
    action="count",
    help="be more verbose; use repeatedly for more info",
)


def _preproc(image_path: Path, info: TensorInfo):
    """Preprocess image for detection model input."""
    batch, height, width, channels = info.unpadded_shape
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Store original dimensions for postprocessing
    orig_h, orig_w = image.shape[:2]
    
    # Resize to model input size
    image = cv2.resize(image, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = image / 255.0
    image = image - np.array(mean)
    image = image / np.array(stddev)
    
    # Quantize according to model requirements
    quantized = np.round(image / info.scale + info.zero_point).clip(-128, 127).astype(np.int8)
    padded = np.pad(quantized, info.padding[1:], mode="constant", constant_values=info.zero_point)
    
    if batch > 1:
        padded = np.repeat(padded[np.newaxis, ...], batch, axis=0)
    
    return padded, (orig_h, orig_w, height, width)


def _non_max_suppression(boxes, scores, class_ids, conf_threshold=0.25, iou_threshold=0.45, max_detections=100):
    """Apply Non-Maximum Suppression to detection results."""
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # First filter by confidence threshold
    valid_mask = scores >= conf_threshold
    if not np.any(valid_mask):
        return np.array([]), np.array([]), np.array([])
    
    filtered_boxes = boxes[valid_mask]
    filtered_scores = scores[valid_mask]
    filtered_class_ids = class_ids[valid_mask]
    
    # Apply NMS per class for better results
    final_boxes = []
    final_scores = []
    final_class_ids = []
    
    unique_classes = np.unique(filtered_class_ids)
    
    for class_id in unique_classes:
        class_mask = filtered_class_ids == class_id
        class_boxes = filtered_boxes[class_mask]
        class_scores = filtered_scores[class_mask]
        
        if len(class_boxes) == 0:
            continue
            
        # Apply NMS for this class
        indices = cv2.dnn.NMSBoxes(
            class_boxes.tolist(), 
            class_scores.tolist(), 
            conf_threshold, 
            iou_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            final_boxes.extend(class_boxes[indices])
            final_scores.extend(class_scores[indices])
            final_class_ids.extend([class_id] * len(indices))
    
    if len(final_boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Convert back to arrays
    final_boxes = np.array(final_boxes)
    final_scores = np.array(final_scores)
    final_class_ids = np.array(final_class_ids)
    
    # Sort by confidence and limit to max detections
    if len(final_scores) > max_detections:
        top_indices = np.argsort(final_scores)[::-1][:max_detections]
        final_boxes = final_boxes[top_indices]
        final_scores = final_scores[top_indices]
        final_class_ids = final_class_ids[top_indices]
    
    return final_boxes, final_scores, final_class_ids


def _postproc_detection(image_path: Path, outputs: list, labels: list[str], 
                       output_infos: list[TensorInfo], image_dims: tuple,
                       conf_threshold: float = 0.25, nms_threshold: float = 0.45, 
                       max_detections: int = 100):
    """
    Postprocess detection outputs to extract bounding boxes, scores, and class IDs.
    
    This function handles YOLO-style output format:
    - Output shape: [batch, num_detections, 85] for COCO (80 classes + 5 for bbox + objectness)
    - Box format: [x_center, y_center, width, height, objectness, class_scores...]
    """
    orig_h, orig_w, model_h, model_w = image_dims
    
    # Dequantize the outputs
    all_boxes = []
    all_scores = []
    all_class_ids = []
    
    for output, info in zip(outputs, output_infos):
        # Dequantize
        out = output[tuple(slice(b, -e if e else None) for b, e in info.padding)]
        out = out.squeeze()
        out = (out.astype(np.float32) - info.zero_point) * info.scale
        
        # Handle different output shapes
        if len(out.shape) == 2:
            # Shape: [num_detections, 85] (batch already squeezed)
            detections = out
        elif len(out.shape) == 3:
            # Shape: [1, num_detections, 85]
            if out.shape[1] > 100:  # Likely a feature map
                LOG.warning(f"Feature map output detected: {out.shape}. This may require anchor-based decoding.")
                continue
            detections = out[0]
        else:
            LOG.warning(f"Unexpected output shape: {out.shape}")
            continue
            
        # Extract components
        if detections.shape[1] < 5:
            LOG.warning(f"Output shape {detections.shape} too small for detection format")
            continue
        
        boxes = detections[:, :4]  # x_center, y_center, width, height
        objectness = detections[:, 4]  # objectness score
        
        if detections.shape[1] > 5:
            class_scores = detections[:, 5:]  # class scores
            
            # Apply sigmoid to raw outputs if they seem to be logits
            if np.any(class_scores > 1) or np.any(class_scores < 0):
                class_scores = 1 / (1 + np.exp(-np.clip(class_scores, -500, 500)))  # sigmoid with clipping
            
            if np.any(objectness > 1) or np.any(objectness < 0):
                objectness = 1 / (1 + np.exp(-np.clip(objectness, -500, 500)))  # sigmoid with clipping
            
            # Combine objectness and class scores
            max_class_scores = np.max(class_scores, axis=1)
            scores = objectness * max_class_scores
            class_ids = np.argmax(class_scores, axis=1)
        else:
            # For models where objectness is the final score
            if np.any(objectness > 1) or np.any(objectness < 0):
                objectness = 1 / (1 + np.exp(-np.clip(objectness, -500, 500)))  # sigmoid with clipping
            scores = objectness
            class_ids = np.zeros(len(scores), dtype=int)  # Single class or unknown
        
        # Filter by confidence threshold
        valid_mask = scores > conf_threshold
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        class_ids = class_ids[valid_mask]
        
        if len(boxes) == 0:
            continue
            
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        
        # Determine coordinate scale and convert to pixel coordinates
        max_coord = np.max(np.abs(boxes[:, :2]))  # Check x,y coordinates
        max_size = np.max(boxes[:, 2:])  # Check width, height
        
        if max_coord <= 1.0 and max_size <= 1.0:
            # Normalized coordinates [0, 1]
            x1 = (x_center - width / 2) * orig_w
            y1 = (y_center - height / 2) * orig_h
            x2 = (x_center + width / 2) * orig_w
            y2 = (y_center + height / 2) * orig_h
        else:
            # Coordinates are in model input scale
            x1_model = x_center - width / 2
            y1_model = y_center - height / 2
            x2_model = x_center + width / 2
            y2_model = y_center + height / 2
            
            # Scale from model input size to original image size
            x1 = x1_model * orig_w / model_w
            y1 = y1_model * orig_h / model_h
            x2 = x2_model * orig_w / model_w
            y2 = y2_model * orig_h / model_h
            
        # Clamp coordinates to image bounds
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        
        # Filter out invalid boxes
        valid_boxes = (x2 > x1) & (y2 > y1)
        if not np.any(valid_boxes):
            continue
            
        x1, y1, x2, y2 = x1[valid_boxes], y1[valid_boxes], x2[valid_boxes], y2[valid_boxes]
        scores = scores[valid_boxes]
        class_ids = class_ids[valid_boxes]
        
        # Create boxes in [x, y, w, h] format for NMS
        widths = x2 - x1
        heights = y2 - y1
        boxes_xywh = np.column_stack([x1, y1, widths, heights])
        
        all_boxes.append(boxes_xywh)
        all_scores.append(scores)
        all_class_ids.append(class_ids)
    
    if not all_boxes:
        print(f"{image_path}: No detections found")
        return
        
    # Combine all outputs
    all_boxes = np.vstack(all_boxes)
    all_scores = np.hstack(all_scores)
    all_class_ids = np.hstack(all_class_ids)
    
    # Apply NMS
    if len(all_boxes) > 0:
        final_boxes, final_scores, final_class_ids = _non_max_suppression(
            all_boxes, all_scores, all_class_ids, conf_threshold, nms_threshold, max_detections
        )
        
        if len(final_boxes) > 0:
            print(f"{image_path}: Found {len(final_boxes)} detections:")
            for i, (box, score, class_id) in enumerate(zip(final_boxes, final_scores, final_class_ids)):
                x, y, w, h = box
                label = labels[int(class_id)] if int(class_id) < len(labels) else f"class_{int(class_id)}"
                print(f"  Detection {i+1}: {label} (confidence: {score:.3f}) "
                      f"bbox: [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}] (pixel coordinates)")
        else:
            print(f"{image_path}: No detections after NMS")
    else:
        print(f"{image_path}: No valid detections found")


def _get_inputs(input_paths: list[Path]) -> collections.abc.Generator[Path, None, None]:
    """Generator for input image paths, supporting both files and directories."""
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(input_path)

    for input_path in input_paths:
        if input_path.is_dir():
            for image_path in input_path.glob("*"):
                if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                    yield image_path
        else:
            yield input_path


class Worker(threading.Thread):
    """Worker thread for model inference."""
    def __init__(self, instance):
        self.instance = instance
        self.inqueue = queue.Queue()
        self.outqueue = queue.Queue()
        super().__init__()
        self.start()

    def run(self):
        while True:
            x = self.inqueue.get()
            if x is None:
                break
            frame_id, inputs, outputs, image_dims = x
            try:
                self.instance.run(inputs, outputs)
            except Exception as e:
                self.outqueue.put(e)
                break
            else:
                self.outqueue.put((frame_id, outputs, image_dims))

    def push(self, frame_id, inputs, outputs, image_dims):
        self.inqueue.put([frame_id, inputs, outputs, image_dims])

    def pop(self):
        x = self.outqueue.get()
        if isinstance(x, Exception):
            raise x
        return x


def run_model(
    model_path: Path,
    aipu_cores: int,
    input_paths: list[Path],
    labels: list[str],
    conf_threshold: float,
    nms_threshold: float,
    max_detections: int,
):
    """Run detection model on input images."""
    with Context() as ctx:
        model = ctx.load_model(model_path)

        input_infos, output_infos = model.inputs(), model.outputs()
        batch_size = input_infos[0].shape[0]

        input_paths = list(_get_inputs(input_paths))
        if len(input_paths) < aipu_cores:
            aipu_cores = len(input_paths)

        num_instances = aipu_cores // batch_size
        if aipu_cores % batch_size:
            LOG.warning(
                f"Number of AIPU cores ({aipu_cores}) is not a multiple of batch size ({batch_size})"
            )

        # Create connections to the AIPU cores
        connections = [ctx.device_connect(None, batch_size) for _ in range(num_instances)]
        LOG.info(f"Creating {num_instances} model instances each with batch size of {batch_size}")
        instances = [
            c.load_model_instance(
                model,
                num_sub_devices=batch_size,
                aipu_cores=batch_size,
            )
            for c in connections
        ]

        # Create input and output buffers for each instance
        inputs = [[np.zeros(t.shape, np.int8) for t in input_infos] for _ in instances]
        outputs = [[np.zeros(t.shape, np.int8) for t in output_infos] for _ in instances]
        workers = [Worker(instance) for instance in instances]

        try:
            prefill = len(workers)
            out_frameno = 0
            
            for in_frameno, image_path in enumerate(input_paths):
                input_data, image_dims = _preproc(image_path, input_infos[0])
                next_available = in_frameno % len(workers)
                inputs[next_available][0][:] = input_data
                workers[next_available].push(
                    image_path, inputs[next_available], outputs[next_available], image_dims
                )
                
                if in_frameno >= prefill:
                    next_ready = out_frameno % len(workers)
                    out_path, outs, dims = workers[next_ready].pop()
                    _postproc_detection(
                        out_path, outs, labels, output_infos, dims, 
                        conf_threshold, nms_threshold, max_detections
                    )
                    out_frameno += 1

            # Drain the remaining workers to extract the last N frames
            for i in range(prefill):
                next_ready = out_frameno % len(workers)
                out_path, outs, dims = workers[next_ready].pop()
                _postproc_detection(
                    out_path, outs, labels, output_infos, dims,
                    conf_threshold, nms_threshold, max_detections
                )
                out_frameno += 1

        finally:
            for worker in workers:
                worker.inqueue.put(None)
            for worker in workers:
                worker.join()


def main(args: argparse.Namespace):
    """Main function."""
    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    desired = levels.get(args.verbose, logging.DEBUG)
    logging.basicConfig(level=desired)

    model_path = Path(args.path)
    
    # Load labels
    labels = []
    if args.labels and args.labels.exists():
        labels = args.labels.read_text().splitlines()
    else:
        # Try COCO labels as fallback
        coco_labels_path = Path(os.path.expandvars("$AXELERA_FRAMEWORK/ax_datasets/labels/coco.names"))
        if coco_labels_path.exists():
            labels = coco_labels_path.read_text().splitlines()
            LOG.info(f"Using COCO labels from: {coco_labels_path}")
        else:
            # Create generic labels
            labels = [f"class_{i}" for i in range(80)]  # Default to 80 classes like COCO
            LOG.warning("No label file found, using generic class names")
    
    if model_path.is_dir():
        model_path /= "model.json"
        
    try:
        run_model(
            model_path,
            args.aipu_cores,
            args.input_paths,
            labels,
            args.conf_threshold,
            args.nms_threshold,
            args.max_detections,
        )
    except Exception as e:
        if args.verbose:
            raise
        print(f'FAIL: {e}')
        return 1
    else:
        return 0


def entrypoint_main():
    """Entry point for the application."""
    args = parser.parse_args()
    try:
        return main(args)
    except RuntimeError as e:
        if args.verbose:
            raise
        return f'ERROR: {e}'


if __name__ == '__main__':
    exit(entrypoint_main())