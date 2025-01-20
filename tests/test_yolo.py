import pytest
import numpy as np
from yolo_minimal_inference.yolo import YOLO
from imageio import imread

# Constants for testing
MODEL_PATH = "tests/static/yolov8n.onnx"
TEST_IMAGE_PATH = "tests/static/zidane.jpg"

@pytest.fixture
def yolo_model():
    """Fixture to initialize the YOLO model."""
    return YOLO(MODEL_PATH, conf_thres=0.5, iou_thres=0.4,is_brg=False)

def test_model_initialization(yolo_model):
    """Test if the YOLO model initializes correctly."""
    assert yolo_model.session is not None, "ONNX Runtime session not initialized."
    assert len(yolo_model.input_names) > 0, "Model input names not detected."
    assert len(yolo_model.output_names) > 0, "Model output names not detected."
    assert yolo_model.input_width > 0 and yolo_model.input_height > 0, "Invalid input dimensions."

def test_prepare_input(yolo_model):
    """Test the image preprocessing step."""
    test_image = imread(TEST_IMAGE_PATH)
    input_tensor = yolo_model.prepare_input(test_image)
    assert input_tensor.shape == (1, 3, yolo_model.input_height, yolo_model.input_width), "Input tensor shape mismatch."
    assert input_tensor.dtype == np.float32, "Input tensor dtype should be float32."
    assert np.max(input_tensor) <= 1.0 and np.min(input_tensor) >= 0.0, "Input tensor values out of range [0, 1]."

def test_inference(yolo_model):
    """Test the inference step with the model."""
    test_image = imread(TEST_IMAGE_PATH)
    input_tensor = yolo_model.prepare_input(test_image)
    outputs = yolo_model.inference(input_tensor)
    assert isinstance(outputs, list), "Model outputs should be a list."
    assert len(outputs) > 0, "No outputs from model inference."
    assert all(isinstance(output, np.ndarray) for output in outputs), "Model outputs should be NumPy arrays."

def test_process_output(yolo_model):
    """Test the output processing step."""
    test_image = imread(TEST_IMAGE_PATH)
    input_tensor = yolo_model.prepare_input(test_image)
    outputs = yolo_model.inference(input_tensor)
    processed_results = yolo_model.process_output(outputs)
    
    # Check that results are in the correct format
    assert isinstance(processed_results.xyxy, np.ndarray), "Bounding boxes should be a NumPy array."
    assert isinstance(processed_results.conf, np.ndarray), "Confidence scores should be a NumPy array."
    assert isinstance(processed_results.cls, np.ndarray), "Class IDs should be a NumPy array."
    
    # Check that results have consistent lengths
    assert len(processed_results.xyxy) == len(processed_results.conf) == len(processed_results.cls), \
        "Output arrays should have consistent lengths."

def test_full_pipeline(yolo_model):
    """Test the complete pipeline from preprocessing to output."""
    test_image = imread(TEST_IMAGE_PATH)
    results = yolo_model(test_image)

    # Check that the final output contains valid detections
    assert isinstance(results.xyxy, np.ndarray), "Bounding boxes should be a NumPy array."
    assert isinstance(results.conf, np.ndarray), "Confidence scores should be a NumPy array."
    assert isinstance(results.cls, np.ndarray), "Class IDs should be a NumPy array."
    assert len(results.xyxy) == len(results.conf) == len(results.cls), "Output arrays should have consistent lengths."
    if len(results.xyxy) > 0:
        assert np.all(results.conf > 0.0), "All confidence scores should be greater than 0."

@pytest.mark.parametrize("image_size", [(320, 320), (1920, 1080), (100, 400)])
def test_different_image_sizes(yolo_model, image_size):
    """Test model with images of different sizes."""
    image = np.random.randint(0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8)
    results = yolo_model(image)
    assert isinstance(results.xyxy, np.ndarray), "Bounding boxes should be a NumPy array."

def test_empty_image(yolo_model):
    """Test model with an empty (all zeros) image."""
    empty_image = np.zeros((640, 640, 3), dtype=np.uint8)
    results = yolo_model(empty_image)
    assert len(results.xyxy) == 0, "No detections expected for an empty image."


@pytest.mark.parametrize("conf_thres, iou_thres", [(0.5, 0.4), (0.8, 0.3), (0.3, 0.6)])
def test_thresholds(yolo_model, conf_thres, iou_thres):
    """Test model behavior with different thresholds."""
    yolo_model.conf_threshold = conf_thres
    yolo_model.iou_threshold = iou_thres
    test_image = imread(TEST_IMAGE_PATH)
    results = yolo_model(test_image)
    if len(results.xyxy) > 0:
        assert np.all(results.conf >= conf_thres), "All confidence scores should meet the threshold."


# def test_execution_providers():
#     """Test model initialization with GPU and CPU execution providers."""
#     yolo_cpu = YOLO(MODEL_PATH, conf_thres=0.5, iou_thres=0.4)
#     assert 'CPUExecutionProvider' in yolo_cpu.session.get_providers(), "CPU provider should be available."

#     if torch.cuda.is_available():
#         yolo_gpu = YOLO(MODEL_PATH)
#         assert 'CUDAExecutionProvider' in yolo_gpu.session.get_providers(), "CUDA provider should be available."


def test_known_image(yolo_model):
    """Test model on an image with known detections."""
    test_image = imread(TEST_IMAGE_PATH)
    results = yolo_model(test_image)
    print(results.cls)
    expected_classes = [0, 0]  # Example class IDs
    assert np.array_equal(results.cls[:len(expected_classes)], expected_classes), "Class IDs don't match expected values."
