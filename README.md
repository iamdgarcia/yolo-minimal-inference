# YOLO Minimal Inference Library

![License](https://img.shields.io/badge/license-MIT-blue)
![Python Version](https://img.shields.io/badge/python-3.11%2B-brightgreen)
![Build Status](https://github.com/iamdgarcia/yolo-minimal-inference/actions/workflows/ci.yml/badge.svg)

**YOLO Minimal Inference Library** is a lightweight Python package designed for efficient and minimal YOLO object detection using ONNX Runtime. This library extracts the essential components for YOLO inference from the Ultralytics library, offering a streamlined alternative for those who need a simple, no-frills solution for YOLO inference.

---

## Features

- **Lightweight**: Focused on essential YOLO inference, reducing overhead.
- **Free Usage**: Open to the community under the MIT license.
- **Fast Inference**: Powered by ONNX Runtime for optimal performance.
- **Flexible Execution**: Supports both CPU and GPU execution providers.
- **Easy Integration**: Simplified API for seamless integration into projects.

---

## Installation

Install the package via pip:

```bash
pip install yolo_minimal_inference
```

---

## Quick Start

### 1. **Download a Pretrained ONNX YOLO Model**

Download YOLO models in ONNX format from:
- [Ultralytics YOLOv5 YOLOv11](https://github.com/ultralytics)

### 2. **Example Usage**

```python
from imageio import imread
from yolo_minimal_inference import YOLO

# Path to the ONNX model
model_path = "path/to/yolov5.onnx"

# Initialize YOLO model
yolo = YOLO(model_path, conf_thres=0.5, iou_thres=0.4,is_bgr=False)

# Load an image
image = imread("path/to/image.jpg")

# Perform inference
results = yolo(image)

# Display results
for box, conf, cls in zip(results.xyxy, results.conf, results.cls):
    print(f"Box: {box}, Confidence: {conf:.2f}, Class: {cls}")
```

---

## TODOs and Progress

This package is under active development. Below is a summary of the work done and the planned next steps:

### **Completed**
- Basic implementation of YOLO inference pipeline:
  - Model initialization with ONNX Runtime.
  - Preprocessing input images (resizing, normalization).
  - Running inference on CPU and GPU (if available).
  - Postprocessing results (Non-Maximum Suppression, confidence filtering).
- Integration with Pytest for unit tests.
- Initial CI/CD setup with GitHub Actions.
- Documentation for installation and usage.

### **Next Steps**
- Add support for batch inference.
- Implement error handling for corrupted or unsupported model files.
- Add a benchmarking utility to measure inference speed on different devices.
- Improve compatibility with more YOLO model variations (e.g., YOLOv8).
- Enhance postprocessing for customizable outputs (e.g., drawing bounding boxes).
- Add visualization utilities to display detections on images/videos.
- Expand test coverage for edge cases:
  - Corrupted images or unsupported formats.
  - Invalid model paths.
  - Custom confidence and IoU thresholds.
- Publish an example notebook showcasing library usage.

If you have suggestions or feature requests, feel free to open an issue in the repository.

---

## API Reference

### **`YOLO` Class**

#### **Initialization**
```python
YOLO(model_path: str, conf_thres: float = 0.5, iou_thres: float = 0.4)
```
- **`model_path`**: Path to the ONNX model file.
- **`conf_thres`**: Confidence threshold for filtering detections.
- **`iou_thres`**: IoU threshold for Non-Maximum Suppression (NMS).

#### **Methods**
1. **`detect_objects(image: np.ndarray) -> Boxes`**
   - Takes an input image, processes it, and returns bounding boxes, confidence scores, and class IDs.

2. **`prepare_input(image: np.ndarray) -> np.ndarray`**
   - Prepares an input image for inference.

3. **`process_output(output: list) -> Boxes`**
   - Post-processes the model output into human-readable results.

---

## Supported Use Cases

- **Lightweight Inference**: Minimal dependencies for object detection.
- **Real-Time Applications**: Efficient enough for live video feeds.
- **Batch Processing**: Analyze multiple images at once (future implementation).

---

## Contributing

Contributions are welcome! Here's how you can get involved:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

---

## Tests

To run tests, clone the repository and execute:

```bash
pytest
```

Ensure you have the required static files (model and test images) in the `tests/static/` directory.

---

## Continuous Integration

This package uses GitHub Actions for CI/CD:
- **Testing**: Runs tests on every push or pull request.
- **Building**: Verifies that the package can be built.
- **Publishing**: Automatically publishes to PyPI on release.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For support or inquiries:
- **Email**: info@iamdgarcia.com
- **GitHub**: [iamdgarcia](https://github.com/iamdgarcia)
- **PyPI**: [YOLO Minimal Inference](https://pypi.org/project/yolo-minimal-inference)

---

## Acknowledgments

Special thanks to the following resources:
- [ONNX Runtime](https://onnxruntime.ai/)
- [Ultralytics YOLO](https://github.com/ultralytics)