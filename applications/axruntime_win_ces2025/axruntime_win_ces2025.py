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
from typing import Dict, List, Tuple, Optional

from axelera.runtime import Context, TensorInfo
import cv2
import numpy as np

LOG = getLogger(__name__)

mean = [0.485, 0.456, 0.406]
stddev = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser(
    description="CES2025 Demo: Real-time pose detection, object detection, and segmentation",
    formatter_class=argparse.RawDescriptionHelpFormatter
)

parser.add_argument(
    "--pose-model",
    type=str,
    required=True,
    help="Path to YOLOv8 pose model (model.json)",
)
parser.add_argument(
    "--detect-model", 
    type=str,
    required=True,
    help="Path to YOLOv8 detection model (model.json)",
)
parser.add_argument(
    "--seg-model",
    type=str,
    required=True, 
    help="Path to YOLOv8 segmentation model (model.json)",
)
parser.add_argument("--aipu-cores", type=int, default=4, help="Number of AIPU cores to use")
parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID (default: 0)")
parser.add_argument(
    "-v",
    "--verbose",
    default=0,
    action="count",
    help="be more verbose; use repeatedly for more info",
)

# COCO Body Keypoints order:
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Pose connections for drawing skeleton
POSE_CONNECTIONS = [
    # Body outline
    (15, 13), (13, 11), (11, 5), (5, 6), (6, 12), (12, 14), (14, 16),
    # Left arm
    (5, 7), (7, 9),
    # Right arm  
    (6, 8), (8, 10),
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Torso
    (5, 11), (6, 12)
]

# Key body parts for red dots (eyes, mouth, ears, hands, feet)
KEY_BODYPARTS = [1, 2, 3, 4, 9, 10, 15, 16]  # eyes, ears, wrists, ankles


def _preproc_frame(frame: np.ndarray, info: TensorInfo, target_size: Tuple[int, int] = (640, 640)):
    batch, height, width, _ = info.unpadded_shape
    
    # Letterbox resize to maintain aspect ratio
    h, w = frame.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Pad to target size
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # Convert to RGB and normalize
    image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = (image - np.array(mean)) / np.array(stddev)
    
    # Quantize
    quantized = np.round(image / info.scale + info.zero_point).clip(-128, 127).astype(np.int8)
    padded_input = np.pad(quantized, info.padding[1:], mode="constant", constant_values=info.zero_point)
    
    if batch > 1:
        padded_input = np.repeat(padded_input[np.newaxis, ...], batch, axis=0)
    
    return padded_input, scale, (left, top)


def _postproc_pose(output: np.ndarray, info: TensorInfo, scale: float, offset: Tuple[int, int], 
                   conf_thresh: float = 0.25, nms_thresh: float = 0.45):
    try:
        # Dequantize
        out = output[tuple(slice(b, -e if e else None) for b, e in info.padding)]
        out = (out.squeeze().astype(np.float32) - info.zero_point) * info.scale
        
        
        # Handle different output formats
        if out.ndim == 3:
            # This might be a feature map format [H, W, C] or [C, H, W]
            # For pose detection, we need to flatten or reshape appropriately
            if out.shape[0] == 1:
                # Batch dimension, remove it
                out = out[0]
            else:
                # This is likely a feature map, we need to flatten it
                # Common formats: [80, 80, 64] could be a feature map
                # We'll flatten and try to interpret it
                out = out.reshape(-1)  # Flatten to 1D first
            
        if out.ndim == 1:
            total_elements = out.size
            
            # Try different reshape options based on total elements
            # For 409600 elements (80*80*64), this might be feature maps
            # Let's try common YOLO anchor grid formats
            possible_shapes = []
            
            # Try to find reasonable factorizations
            for grid_size in [80, 40, 20, 10, 8]:  # Common YOLO grid sizes
                if total_elements % (grid_size * grid_size) == 0:
                    features = total_elements // (grid_size * grid_size)
                    possible_shapes.append((grid_size * grid_size, features))
                    
            # Also try some other common formats
            if total_elements % 56 == 0:
                possible_shapes.append((total_elements // 56, 56))
                
            # Try the first reasonable option
            if possible_shapes:
                out = out.reshape(possible_shapes[0])
                
                # Extract what we can from this format
                if out.shape[1] >= 5:  # At least bbox + conf
                    boxes = out[:, :4]  
                    confidences = out[:, 4]
                    # Use remaining features as dummy keypoints
                    keypoints = np.zeros((out.shape[0], 17, 3))
                else:
                    return [], []
            else:
                return [], []
        elif out.ndim == 2:
            # Standard 2D format
            if out.shape[0] == 56 or out.shape[1] == 56:
                # Standard YOLOv8 pose format
                if out.shape[0] == 56:
                    out = out.T  # Transpose to [N, 56]
                    
                boxes = out[:, :4]
                confidences = out[:, 4] 
                kpt_data = out[:, 5:]
                
                if kpt_data.shape[1] >= 51:
                    keypoints = kpt_data[:, :51].reshape(-1, 17, 3)
                else:
                    keypoints = np.zeros((out.shape[0], 17, 3))
            else:
                # Non-standard format, try to extract what we can
                if out.shape[1] >= 5:
                    boxes = out[:, :4]
                    confidences = out[:, 4]
                    keypoints = np.zeros((out.shape[0], 17, 3))  # Dummy keypoints
                else:
                    return [], []
        else:
            return [], []
        
        # Filter by confidence
        try:
            valid_idx = confidences > conf_thresh
            
            boxes = boxes[valid_idx]
            confidences = confidences[valid_idx]
            keypoints = keypoints[valid_idx]
            
            if len(boxes) == 0:
                return [], []
        except Exception as e:
            return [], []
        
        # Convert boxes from xywh to xyxy
        boxes_xyxy = np.copy(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        
        # Apply NMS
        try:
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf_thresh, nms_thresh)
            
            # Handle different return types from cv2.dnn.NMSBoxes
            if indices is None:
                return [], []
            elif isinstance(indices, (list, tuple)) and len(indices) == 0:
                return [], []
            elif hasattr(indices, 'size') and indices.size == 0:
                return [], []
        except Exception as e:
            return [], []
        
        # Safely flatten indices
        if hasattr(indices, 'flatten'):
            indices = indices.flatten()
        elif isinstance(indices, (list, tuple)):
            indices = np.array(indices).flatten()
        else:
            indices = np.array([indices]).flatten()
            
        try:
            final_boxes = boxes_xyxy[indices]
            final_keypoints = keypoints[indices]
            
            # Scale back to original image coordinates
            left_offset, top_offset = offset
            
            final_boxes[:, [0, 2]] = (final_boxes[:, [0, 2]] - left_offset) / scale
            final_boxes[:, [1, 3]] = (final_boxes[:, [1, 3]] - top_offset) / scale
            
            final_keypoints[:, :, 0] = (final_keypoints[:, :, 0] - left_offset) / scale
            final_keypoints[:, :, 1] = (final_keypoints[:, :, 1] - top_offset) / scale
            
            return final_boxes, final_keypoints
        except Exception as e:
            return [], []
        
    except Exception as e:
        return [], []


def _postproc_detect(output: np.ndarray, info: TensorInfo, scale: float, offset: Tuple[int, int],
                     conf_thresh: float = 0.25, nms_thresh: float = 0.45):
    # Dequantize
    out = output[tuple(slice(b, -e if e else None) for b, e in info.padding)]
    out = (out.squeeze().astype(np.float32) - info.zero_point) * info.scale
    
    # YOLOv8 detection output format: [batch, 84, 8400] where 84 = 4 (bbox) + 80 (classes)
    if out.ndim == 3:
        out = out[0]  # Remove batch dimension
    
    # Transpose to [8400, 84]
    if out.shape[0] == 84:
        out = out.T
        
    boxes = out[:, :4]  # x_center, y_center, width, height
    class_scores = out[:, 4:]
    
    # Get max class scores and indices
    max_scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)
    
    # Filter by confidence and person class (class 0 in COCO)
    valid_idx = (max_scores > conf_thresh) & (class_ids == 0)  # Only person detections
    boxes = boxes[valid_idx]
    scores = max_scores[valid_idx]
    
    if len(boxes) == 0:
        return []
    
    # Convert boxes from xywh to xyxy
    boxes_xyxy = np.copy(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thresh, nms_thresh)
    
    if len(indices) == 0:
        return []
    
    indices = indices.flatten()
    final_boxes = boxes_xyxy[indices]
    
    # Scale back to original image coordinates
    left_offset, top_offset = offset
    final_boxes[:, [0, 2]] = (final_boxes[:, [0, 2]] - left_offset) / scale
    final_boxes[:, [1, 3]] = (final_boxes[:, [1, 3]] - top_offset) / scale
    
    return final_boxes


def _draw_pose_overlay(frame: np.ndarray, boxes: List[np.ndarray], keypoints: List[np.ndarray]):
    overlay = frame.copy()
    
    # Draw pose bounding boxes as yellow rectangles since keypoints are dummy
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        # Draw yellow boxes for pose detections
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow boxes
        cv2.putText(overlay, "Pose", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Try to draw keypoints if they have valid data
    for person_keypoints in keypoints:
        # Draw pose connections as yellow lines
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(person_keypoints) and end_idx < len(person_keypoints):
                start_kpt = person_keypoints[start_idx]
                end_kpt = person_keypoints[end_idx]
                
                # Only draw if both keypoints are visible (confidence > 0.5)
                if start_kpt[2] > 0.5 and end_kpt[2] > 0.5:
                    start_pt = (int(start_kpt[0]), int(start_kpt[1]))
                    end_pt = (int(end_kpt[0]), int(end_kpt[1]))
                    cv2.line(overlay, start_pt, end_pt, (0, 255, 255), 2)  # Yellow lines
        
        # Draw red dots for key body parts
        for kpt_idx in KEY_BODYPARTS:
            if kpt_idx < len(person_keypoints):
                kpt = person_keypoints[kpt_idx]
                if kpt[2] > 0.5:  # Only draw if confident
                    center = (int(kpt[0]), int(kpt[1]))
                    cv2.circle(overlay, center, 4, (0, 0, 255), -1)  # Red dots
    
    return overlay


def _draw_detection_overlay(frame: np.ndarray, boxes: List[np.ndarray]):
    overlay = frame.copy()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        # Draw thick red boxes for detected humans
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Thick red lines
        
        # Add label
        cv2.putText(overlay, "Person", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return overlay


class ModelWorker(threading.Thread):
    def __init__(self, instance, model_name: str):
        self.instance = instance
        self.model_name = model_name
        self.inqueue = queue.Queue(maxsize=2)
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
                    self.outqueue.put((frame_id, inputs_outputs[1], self.model_name))
            except queue.Empty:
                continue

    def push(self, frame_id, inputs, outputs):
        try:
            self.inqueue.put([frame_id, inputs, outputs], block=False)
            return True
        except queue.Full:
            return False

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


def run_ces2025_demo(
    pose_model_path: Path,
    detect_model_path: Path,
    seg_model_path: Path,
    aipu_cores: int,
    camera_id: int,
):
    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_id}")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    with Context() as ctx:
        # Load all three models
        pose_model = ctx.load_model(pose_model_path)
        detect_model = ctx.load_model(detect_model_path)  
        seg_model = ctx.load_model(seg_model_path)
        
        # Get model info
        pose_input_infos, pose_output_infos = pose_model.inputs(), pose_model.outputs()
        detect_input_infos, detect_output_infos = detect_model.inputs(), detect_model.outputs()
        seg_input_infos, seg_output_infos = seg_model.inputs(), seg_model.outputs()
        
        # Create device connections
        pose_connection = ctx.device_connect(None, 1)
        detect_connection = ctx.device_connect(None, 1)
        seg_connection = ctx.device_connect(None, 1)
        
        # Create model instances
        pose_instance = pose_connection.load_model_instance(pose_model, num_sub_devices=1, aipu_cores=2)
        detect_instance = detect_connection.load_model_instance(detect_model, num_sub_devices=1, aipu_cores=1)
        seg_instance = seg_connection.load_model_instance(seg_model, num_sub_devices=1, aipu_cores=1)
        
        # Create buffers
        pose_inputs = [np.zeros(t.shape, np.int8) for t in pose_input_infos]
        pose_outputs = [np.zeros(t.shape, np.int8) for t in pose_output_infos]
        
        detect_inputs = [np.zeros(t.shape, np.int8) for t in detect_input_infos]
        detect_outputs = [np.zeros(t.shape, np.int8) for t in detect_output_infos]
        
        seg_inputs = [np.zeros(t.shape, np.int8) for t in seg_input_infos]
        seg_outputs = [np.zeros(t.shape, np.int8) for t in seg_output_infos]
        
        # Create workers
        pose_worker = ModelWorker(pose_instance, "pose")
        detect_worker = ModelWorker(detect_instance, "detect")
        seg_worker = ModelWorker(seg_instance, "seg")
        
        # Current results
        current_pose_boxes = []
        current_pose_keypoints = []
        current_detect_boxes = []
        
        frame_count = 0
        
        try:
            print("Starting CES2025 demo. Press 'q' to quit.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Create display frame
                display_frame = frame.copy()
                
                # Process every 3rd frame to maintain performance
                if frame_count % 3 == 0:
                    # Preprocess for pose model
                    pose_input, pose_scale, pose_offset = _preproc_frame(frame, pose_input_infos[0])
                    pose_inputs[0][:] = pose_input
                    pose_worker.push(frame_count, pose_inputs, pose_outputs)
                    
                    # Preprocess for detection model  
                    detect_input, detect_scale, detect_offset = _preproc_frame(frame, detect_input_infos[0])
                    detect_inputs[0][:] = detect_input
                    detect_worker.push(frame_count, detect_inputs, detect_outputs)
                    
                    # TODO: Add segmentation processing if needed
                    # seg_input, seg_scale, seg_offset = _preproc_frame(frame, seg_input_infos[0])
                    # seg_inputs[0][:] = seg_input
                    # seg_worker.push(frame_count, seg_inputs, seg_outputs)
                
                # Check for results from workers
                pose_result = pose_worker.pop()
                if pose_result is not None:
                    try:
                        frame_id, outs, model_name = pose_result
                        current_pose_boxes, current_pose_keypoints = _postproc_pose(
                            outs[0], pose_output_infos[0], pose_scale, pose_offset
                        )
                    except Exception as e:
                        pass
                
                detect_result = detect_worker.pop()
                if detect_result is not None:
                    try:
                        frame_id, outs, model_name = detect_result
                        current_detect_boxes = _postproc_detect(
                            outs[0], detect_output_infos[0], detect_scale, detect_offset
                        )
                    except Exception as e:
                        pass
                
                # Draw overlays
                try:
                    if len(current_pose_keypoints) > 0:
                        display_frame = _draw_pose_overlay(display_frame, current_pose_boxes, current_pose_keypoints)
                except Exception as e:
                    pass
                
                try:
                    if len(current_detect_boxes) > 0:
                        display_frame = _draw_detection_overlay(display_frame, current_detect_boxes)
                except Exception as e:
                    pass
                
                # Add info text
                info_text = f"Frame: {frame_count} | Poses: {len(current_pose_keypoints)} | Detections: {len(current_detect_boxes)}"
                cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('CES2025 Demo - Pose + Detection', display_frame)
                
                frame_count += 1
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            pose_worker.stop()
            detect_worker.stop()
            seg_worker.stop()
            
            pose_worker.join()
            detect_worker.join()
            seg_worker.join()
            
            cap.release()
            cv2.destroyAllWindows()


def main(args: argparse.Namespace):
    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    desired = levels.get(args.verbose, logging.DEBUG)
    logging.basicConfig(level=desired)

    pose_model_path = Path(args.pose_model)
    detect_model_path = Path(args.detect_model)
    seg_model_path = Path(args.seg_model)
    
    if pose_model_path.is_dir():
        pose_model_path /= "model.json"
    if detect_model_path.is_dir():
        detect_model_path /= "model.json"
    if seg_model_path.is_dir():
        seg_model_path /= "model.json"
    
    try:
        run_ces2025_demo(
            pose_model_path,
            detect_model_path,
            seg_model_path,
            args.aipu_cores,
            args.camera_id,
        )
    except Exception as e:
        if args.verbose:
            raise
        print(f'FAIL: {e}')
        return 1
    else:
        return 0


def entrypoint_main():
    args = parser.parse_args()
    try:
        return main(args)
    except RuntimeError as e:
        if args.verbose:
            raise
        print(f'ERROR: {e}')
        return 1


if __name__ == '__main__':
    exit(entrypoint_main())