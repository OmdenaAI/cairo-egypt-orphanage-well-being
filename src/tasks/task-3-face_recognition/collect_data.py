import numpy as np
import cv2
from ultralytics import YOLO
from utils.deep_sort.deep_sort import DeepSort
import os

deepsort_tracker = DeepSort("project_assets/models/ckpt.t7", use_cuda=True)
face_detector = YOLO("project_assets/models/yolov8n-face.pt")
def detect_people(frame):
    """
    Detect people in a frame and update tracking information.

    Args:
        frame: Input frame for object detection and tracking.

    Returns:
        results: Detection and tracking results.
    """
    results = face_detector(frame, verbose=False)
    deepsort_outputs = []

    tracked_objects = deepsort_tracker.deepsort_update(
        results[0].boxes.conf,
        results[0].boxes.cls,
        results[0].boxes.xywh,  # Object detections for the current frame
        results[0].orig_img  # Image features
    )

    # If no tracks were updated or created, create an empty array
    # (with shape (0, 8) to signify this)
    if len(tracked_objects) == 0:
        tracked_objects = np.ones((0, 8))

    # Append the output of DeepSort for this object to our list of outputs
    deepsort_outputs.append(tracked_objects.astype(np.float32))

    # Replace the YOLO predictions with the updated DeepSort outputs.
    # This means we're now working with the objects as they've been tracked
    # across frames, rather than as they were initially detected in the current frame.
    results[0].boxes = deepsort_outputs
    return results


def extract_faces(results):
    for box in results[0].boxes[0]:
        # Extract bounding box coordinates and track ID
        x1, y1, x2, y2 = box[0:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = str(int(box[5]))
        face_region = frame[y1:y2, x1:x2]
        save_dir = track_id

        # Check if the directory already exists, create it if not
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # Generate a unique filename
        img_filename = 'img.jpg'
        img_counter = 0
        while os.path.exists(os.path.join(save_dir, img_filename)):
            img_counter += 1
            img_filename = f'img{img_counter}.jpg'

        # Save the face image with the unique filename
        cv2.imwrite(os.path.join(save_dir, img_filename), face_region)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (247, 216, 0), 2)


if __name__ == "__main__":
    src = "project_assets/videos/Video2_chunk_2.mp4"
    cam = cv2.VideoCapture(src)
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab a frame, try again")
            break
        results = detect_people(frame)
        # extract_faces(results)
        cv2.imshow("IMG", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC
            print('Esc pressed, closing...')
            break

    cam.release()
    cv2.destroyAllWindows()
