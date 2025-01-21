import numpy as np

def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list:
    """
    Perform Non-Maximum Suppression (NMS) on bounding boxes.

    Args:
        boxes (np.ndarray): Array of bounding boxes in (x1, y1, x2, y2) format.
        scores (np.ndarray): Confidence scores for each bounding box.
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        list: Indices of the boxes to keep after NMS.
    """
    # Sort boxes by confidence scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []

    while sorted_indices.size > 0:
        # Select the box with the highest score and add its index to keep_boxes
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the selected box with the rest
        ious = compute_iou(boxes[box_id], boxes[sorted_indices[1:]])

        # Filter out boxes with IoU above the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # Update the sorted indices (shift by 1 to exclude the picked box)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute the Intersection over Union (IoU) between a box and a set of boxes.

    Args:
        box (np.ndarray): Single bounding box in (x1, y1, x2, y2) format.
        boxes (np.ndarray): Array of bounding boxes in (x1, y1, x2, y2) format.

    Returns:
        np.ndarray: Array of IoU values.
    """
    # Compute coordinates of the intersection rectangle
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute areas of individual boxes
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Compute union area
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area
    return iou

def xywh2xyxy(bboxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from (x, y, w, h) format to (x1, y1, x2, y2) format.

    Args:
        bboxes (np.ndarray): Array of bounding boxes in (x, y, w, h) format.

    Returns:
        np.ndarray: Array of bounding boxes in (x1, y1, x2, y2) format.
    """
    converted_bboxes = np.copy(bboxes)
    converted_bboxes[..., 0] = bboxes[..., 0] - bboxes[..., 2] / 2  # x1
    converted_bboxes[..., 1] = bboxes[..., 1] - bboxes[..., 3] / 2  # y1
    converted_bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2] / 2  # x2
    converted_bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3] / 2  # y2
    return converted_bboxes
