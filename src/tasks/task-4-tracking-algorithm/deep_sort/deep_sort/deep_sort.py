# Required packages and modules
import numpy as np
import torch
from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker
import cv2

# Specify the classes available from this module when imported
__all__ = ['DeepSort']


class DeepSort(object):
    """
    The DeepSort class is responsible for sorting and tracking objects
    based on appearance and motion.

    Attributes:
    - min_confidence: Minimum confidence for valid detection
    - nms_max_overlap: Overlapping threshold for non-maximum suppression
    - use_appearence: Flag to decide if appearance features are used
    - extractor: Feature extraction model
    - tracker: Object tracker instance
    """

    def __init__(self, model_path="deepsort_checkpoint.t7", max_dist=0.2, min_confidence=0.3,
                 nms_max_overlap=3.0, max_iou_distance=0.7,
                 max_age=70, n_init=2, nn_budget=100,
                 use_cuda=True, use_appearence=True):
        """
        Constructor for DeepSort class.

        Args:
        - model_path: Path to the model for feature extraction
        - max_dist: Maximum distance for matching
        - min_confidence: Minimum confidence for valid detection
        - nms_max_overlap: Overlapping threshold for non-maximum suppression
        - max_iou_distance: Maximum IoU distance for detection
        - max_age: Maximum age for a track before removal
        - n_init: Number of initializations
        - nn_budget: Budget for nearest neighbor
        - use_cuda: Flag to use GPU if available
        - use_appearence: Flag to use appearance feature
        """

        # Initialize the attributes
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.use_appearence = use_appearence

        # Extractor for appearance features
        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        # Initialize the metric for nearest neighbor matching
        metric = NearestNeighborDistanceMetric("cosine", max_dist, nn_budget)

        # Tracker to track multiple objects
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, labels, ori_img):
        """
        Update tracks based on detections.

        Args:
        - bbox_xywh: Bounding boxes in format (x_center, y_center, width, height)
        - confidences: Confidence scores for each bounding box
        - labels: Object labels for each bounding box
        - ori_img: Original image

        Returns:
        - outputs: Array containing bounding boxes, labels, track IDs and velocities
        """
        bbox_xywh = bbox_xywh.cpu()
        confidences = confidences.cpu()
        labels = labels.cpu()
        # Get the image dimensions
        self.height, self.width = ori_img.shape[:2]

        # Extract features from the image if appearance is used
        if self.use_appearence:
            features = self._get_features(bbox_xywh, ori_img)
        else:
            features = np.array([np.array([0.5, 0.5]) for _ in range(len(bbox_xywh))])

        # Convert from center format to top-left format
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)

        # Create detections based on bounding boxes, confidence, labels, and features
        detections = [Detection(bbox_tlwh[i], conf, labels[i], features[i]) for i, conf in enumerate(confidences) if
                      conf > self.min_confidence]

        # Predict the next state for all tracks
        self.tracker.predict()

        # Update the tracker based on detections
        self.tracker.update(detections)

        # Retrieve and format the track outputs
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            outputs.append(
                np.array([x1, y1, x2, y2, track.label, track.track_id, 10 * track.mean[4], 10 * track.mean[5]],
                         dtype=np.int32))

        if outputs:
            outputs = np.stack(outputs, axis=0)

        return outputs

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        """
        Convert bounding box format from center (x_center, y_center, width, height) to top-left corner (x_top_left, y_top_left, width, height).

        Args:
        - bbox_xywh: Bounding box in center format.

        Returns:
        - bbox_tlwh: Bounding box in top-left corner format.
        """

        # If the bounding box is a numpy array, create a copy of it
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        # If the bounding box is a torch tensor, clone it
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()

        # Compute top-left x-coordinate
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        # Compute top-left y-coordinate
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.

        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        """
        Convert bounding box format from center (x_center, y_center, width, height) to two corner points (x1, y1, x2, y2).

        Args:
        - bbox_xywh: Bounding box in center format.

        Returns:
        - Tuple containing two corner points (x1, y1, x2, y2).
        """

        # Extract individual components of the bounding box
        x, y, w, h = bbox_xywh

        # Calculate the two corner coordinates using the center format details
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)

        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        Convert bounding box format from top-left corner (x_top_left, y_top_left, width, height) to two corner points (x1, y1, x2, y2).

        Args:
        - bbox_tlwh: Bounding box in top-left corner format.

        Returns:
        - Tuple containing two corner points (x1, y1, x2, y2).
        """

        # Extract individual components of the bounding box
        x, y, w, h = bbox_tlwh

        # Convert top-left to two corners format
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)

        return x1, y1, x2, y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        """
        Convert bounding box format from two corner points (x1, y1, x2, y2) to top-left corner (x_top_left, y_top_left, width, height).

        Args:
        - bbox_xyxy: Tuple containing two corner points.

        Returns:
        - Bounding box in top-left corner format.
        """

        # Extract corner points
        x1, y1, x2, y2 = bbox_xyxy

        # Compute the top-left corner coordinates and width, height using the two corner points
        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)

        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        """
        Extract appearance features for each bounding box.

        Args:
        - bbox_xywh: Bounding boxes in the format (x_center, y_center, width, height)
        - ori_img: Original image

        Returns:
        - features: Array of appearance features for each bounding box
        """

        # List to store cropped images for each bounding box
        im_crops = []

        # For each bounding box, convert it to corner format and extract the image portion
        for box in bbox_xywh:
            # Convert bounding box format from center to two corner points format
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)

            # Extract the image portion corresponding to the bounding box
            im = ori_img[y1:y2, x1:x2]

            # Append the cropped image to the list
            im_crops.append(im)

        # If there are cropped images, extract features for each of them using the extractor model
        if im_crops:
            features = self.extractor(im_crops)
        else:
            # If no cropped images are present, return an empty array
            features = np.array([])

        return features

    def deepsort_update(self, confidence_scores, class_labels, xywh_bboxes, np_img):
        """
        Update the DeepSort tracker with new data.

        Args:
            confidence_scores (list): Confidence scores associated with the tracked objects.
            class_labels (list): Class labels of the tracked objects.
            xywh_bboxes (list): List of bounding box coordinates [x, y, width, height].
            np_img (numpy.ndarray): The input image as a NumPy array.

        Returns:
            list: Updated tracker outputs, including tracked object information.
        """
        # Convert the NumPy image array from BGR to RGB format using OpenCV
        rgb_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

        # Update the DeepSort tracker with the provided data
        outputs = self.update(xywh_bboxes, confidence_scores, class_labels, rgb_img)

        return outputs