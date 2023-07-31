"""
This code is just a starting point and not meant to be used for production
"""

import cv2
import math
import numpy as np
from ultralytics import YOLO
from seaborn import color_palette


def load_class_names(file_name):
    """
    Returns a list of class names read from the file `file_name`.

    Args:
        file_name (str): The path to the file containing the class names.

    Returns:
        List[str]: A list of class names.
    """

    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()

    return class_names


def draw_bbox(frame, boxes, class_names, colors):
    """
    Draws bounding boxes with labels on the input frame.

    Args:
        frame (numpy.ndarray): The input image frame.
        boxes (List[Object]): List of bounding boxes.
        class_names (List[str]): List of class names.
        colors (List[Tuple[int]]): List of RGB color values.

    Returns:
        None
    """
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Extracting the class label and name
        cls = int(box.cls[0])

        class_name = class_names[cls]

        # Retrieving the color for the class
        color = colors[cls]
        # Drawing the bounding box on the image
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Formatting the confidence level and label text
        conf = math.ceil((box.conf[0] * 100)) / 100
        label = f'{class_name} ({conf}%)'

        # Calculating the size of the label text
        text_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
        # Calculating the coordinates for the background rectangle of the label
        rect_coords = x1 + text_size[0], y1 - text_size[1] - 3

        # Drawing the background rectangle and the label text
        cv2.rectangle(frame, (x1, y1), rect_coords, color, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, y1 - 2), 0, 1, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)


def run_yolo(model_name='yolo_assets/Models/yolov8s-face.pt', source=0,
             class_path="yolo_assets/Classes/Face.txt"):
    """
    Performs object detection on an image or video.

    Args:
        model_name (str): The name of the model to use for object detection. Default is 'yolov8s.pt'.
        source (Union[str, int]): The path to the image or video file or webcam index. Default is 0 (webcam).
        class_path (str): The path to the file containing class names. Default is 'classes.txt'.
    """
    # Initializing the YOLO model
    model = YOLO(model_name)
    # Loading the class names from the file
    class_names = load_class_names(class_path)
    n_classes = len(class_names)
    #  Generating colors for each class
    colors = {}
    #  Generate a color palette
    for i in range(n_classes):
        color = tuple((np.array(color_palette('hls', n_classes)) * 255)[i])
        colors[i] = colo
    # Capturing the video from the source
    cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop when the video ends

        results = model(frame, stream=True, conf=0.5, verbose=False)

        for i, result in enumerate(results):
            boxes = result.boxes
            draw_bbox(frame, boxes, class_names, colors)

        cv2.imshow("Image", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Release the video capture
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_yolo(source='yolo_assets/Data/video.mp4')
