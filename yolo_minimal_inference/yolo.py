import time
from imageio import imread
import numpy as np
import onnxruntime
from yolo_minimal_inference.utils import xywh2xyxy, nms
from yolo_minimal_inference import cv
import logging
class Boxes:
    xyxy: np.array = []
    conf : np.array = []
    cls : np.array = []



class YOLO:
    input_width : int = 640
    input_height : int = 640
    is_bgr : bool = True
    verbose: bool = False
    preproc_time : float = 0
    inference_time : float = 0
    postproc_time : float = 0

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5,is_brg=False,verbose=False):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.is_bgr = is_brg
        self.verbose = verbose
        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        if isinstance(image, str):
            image = imread(image)
        return self.detect_objects(image)


    def initialize_model(self, path):
        providers = ['CPUExecutionProvider']
        #TODO only add torch if GPU
        # if torch.cuda.is_available():
        #     providers.insert(0, 'CUDAExecutionProvider')
        self.session = onnxruntime.InferenceSession(path,providers=providers)
        # self.is_quantized = self.model_is_quantized()        # Get model info
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        if self.verbose:
            start = time.perf_counter()

        input_tensor = self.prepare_input(image)
        if self.verbose:
            self.preproc_time =(time.perf_counter() - start)*1000
            start = time.perf_counter()

        outputs = self.inference(input_tensor)

        if self.verbose:
            self.inference_time =(time.perf_counter() - start)*1000
            start = time.perf_counter()

        proc_output = self.process_output(outputs)
        if self.verbose:
            self.postproc_time =(time.perf_counter() - start)*1000
            logging.info(f"Execution time: Preproc: {self.preproc_time:.2f} ms Inference: {self.inference_time:.2f} ms Postproc: {self.postproc_time:.2f} ms")

        return  proc_output

    
    
    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        if self.is_bgr:
            input_img = image[:, :, ::-1]
        else:
            input_img = image
        # Resize input image
        input_img = cv.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor


    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        return outputs

    def process_output(self, output):
        results = Boxes()

        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        results.xyxy =  boxes[indices]
        results.conf = scores[indices]
        results.cls = class_ids[indices]
        return results

    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes)
        boxes = xywh2xyxy(boxes)
        return boxes

    def rescale_boxes(self, boxes):
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]