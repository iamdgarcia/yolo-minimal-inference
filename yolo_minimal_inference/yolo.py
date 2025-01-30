import time
import logging
from imageio import imread
import numpy as np
import onnxruntime
from yolo_minimal_inference.utils import xywh2xyxy, nms
from yolo_minimal_inference import cv


class Boxes:
    """Container for YOLO detection results."""
    xyxy: np.ndarray = []  # Bounding boxes in xyxy format
    conf: np.ndarray = []  # Confidence scores
    cls: np.ndarray = []  # Class IDs


class YOLO:
    """
    Minimal YOLO inference pipeline using ONNX Runtime.

    Args:
        path (str): Path to the ONNX model file.
        conf_thres (float): Confidence threshold for filtering detections.
        iou_thres (float): IoU threshold for Non-Maximum Suppression.
        is_bgr (bool): Whether input images are in BGR format.
        verbose (bool): Enable logging for processing times.
    """

    def __init__(self, path: str, conf_thres: float = 0.7, iou_thres: float = 0.5, is_bgr: bool = False, verbose: bool = False):
        self.input_width: int = 640
        self.input_height: int = 640
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.is_bgr = is_bgr
        self.verbose = verbose
        self.preproc_time: float = 0
        self.inference_time: float = 0
        self.postproc_time: float = 0

        self.session = None
        self.input_names = []
        self.output_names = []

        self.initialize_model(path)

    def __call__(self, image):
        """
        Perform inference on a given image.

        Args:
            image (str or np.ndarray): Path to the image or the image array.

        Returns:
            Boxes: Processed YOLO detections.
        """
        if isinstance(image, str):
            image = imread(image, pilmode='RGB')
        return self.detect_objects(image)

    def initialize_model(self, path: str):
        """Initialize the ONNX model session."""
        providers = ['CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(path, providers=providers)
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image: np.ndarray) -> Boxes:
        """Detect objects in the input image."""
        if self.verbose:
            start = time.perf_counter()

        input_tensor = self.prepare_input(image)

        if self.verbose:
            self.preproc_time = (time.perf_counter() - start) * 1000
            start = time.perf_counter()

        outputs = self.inference(input_tensor)

        if self.verbose:
            self.inference_time = (time.perf_counter() - start) * 1000
            start = time.perf_counter()

        results = self.process_output(outputs)

        if self.verbose:
            self.postproc_time = (time.perf_counter() - start) * 1000
            logging.info(
                f"Execution time: Preprocessing: {self.preproc_time:.2f} ms, "
                f"Inference: {self.inference_time:.2f} ms, "
                f"Postprocessing: {self.postproc_time:.2f} ms"
            )

        return results

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        """Prepare the input image for the YOLO model."""
        self.img_height, self.img_width = image.shape[:2]

        # Convert BGR to RGB if necessary
        if self.is_bgr:
            image = image[:, :, ::-1]

        # Resize and normalize the image
        input_img = cv.resize(image, (self.input_width, self.input_height))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)  # Convert to channel-first

        return input_img[np.newaxis, :, :, :].astype(np.float32)

    def inference(self, input_tensor: np.ndarray) -> list:
        """Run inference on the input tensor."""
        return self.session.run(self.output_names, {self.input_names[0]: input_tensor})

    def process_output(self, output: list) -> Boxes:
        """Process the raw model output into structured detections."""
        results = Boxes()
        predictions = np.squeeze(output[0]).T

        # Filter by confidence threshold
        scores = np.max(predictions[:, 4:], axis=1)
        valid_indices = scores > self.conf_threshold
        predictions = predictions[valid_indices]
        scores = scores[valid_indices]

        # Get class IDs and bounding boxes
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = self.extract_boxes(predictions)

        # Apply Non-Maximum Suppression
        indices = nms(boxes, scores, self.iou_threshold)

        results.xyxy = boxes[indices]
        results.conf = scores[indices]
        results.cls = class_ids[indices]
        return results

    def extract_boxes(self, predictions: np.ndarray) -> np.ndarray:
        """Extract and rescale bounding boxes."""
        boxes = predictions[:, :4]
        return xywh2xyxy(self.rescale_boxes(boxes))

    def rescale_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """Rescale bounding boxes to the original image dimensions."""
        input_shape = np.array([self.input_width, self.input_height] * 2)
        scale_factors = np.array([self.img_width, self.img_height] * 2)
        return boxes / input_shape * scale_factors

    def get_input_details(self):
        """Retrieve input details from the ONNX model."""
        model_inputs = self.session.get_inputs()
        self.input_names = [input.name for input in model_inputs]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        """Retrieve output details from the ONNX model."""
        model_outputs = self.session.get_outputs()
        self.output_names = [output.name for output in model_outputs]
