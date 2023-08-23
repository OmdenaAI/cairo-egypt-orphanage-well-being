import cv2
from ultralytics import YOLO

className = ["Face"]
#video_out_path = "videos/output_video.mp4"
class FaceDetector:
    def __init__(self, yolo_weights, conf_threshold=0.5, width=1280, height=720):
        """
        Initialize the FaceDetector.

        Parameters:
            yolo_weights (str): Path to the YOLO weights file.
            conf_threshold (float, optional): Confidence threshold for detection. Defaults to 0.5.
            width (int, optional): Frame width. Defaults to 1280.
            height (int, optional): Frame height. Defaults to 720.
        """
        self.model = YOLO(yolo_weights)
        self.cap = cv2.VideoCapture()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.conf_threshold = conf_threshold

    def detect_faces(self, video_source=0):
        """
        Detect faces in a video stream (live-streaming or webcam) and display bounding boxes.

        Parameters:
            video_source (int, optional): Index of the video source (e.g., 0 for webcam). Defaults to 0.
        """
        self.cap.open(video_source)


        try:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    break

                detections = self.model(frame, stream=True)

                # get b_box, confidence, and class for each detected object
                for detection in detections:
                    bounding_boxes = detection.boxes
                    for box in bounding_boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf[0]
                        cls = int(box.cls[0])

                        if conf > self.conf_threshold and cls == 0:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                            text_size, _ = cv2.getTextSize(f'{className[cls]} {conf:.2f}', cv2.FONT_HERSHEY_PLAIN, 1, 2)
                            text_width, text_height = text_size

                            if y1 - 30 < 0:
                                y1 = 30
                            if x1 + text_width + 5 > frame.shape[1]:
                                x1 = frame.shape[1] - text_width - 5
                            if y1 - text_height - 10 < 0:
                                y1 = text_height + 10

                            cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_width + 5, y1 - 5),
                                          (0, 255, 0), -1)
                            cv2.putText(frame, f'{className[cls]} {conf:.2f}', (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                #cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), self.cap.get(cv2.CAP_PROP_FPS),
                                          #(frame.shape[1], frame.shape[0]))
                #cap_out.write(frame)

                cv2.imshow('Face', frame)

                key = cv2.waitKey(1)
                if key == ord('q') or key == 27:
                    break

        except Exception as e:
            print(f"Error occurred: {e}")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

def main():
    yolo_weights = "yolov8_weights/face.pt"

    face_detector = FaceDetector(yolo_weights, conf_threshold=0.1, width=800, height=620)

    try:
        face_detector.detect_faces()
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping...")
    finally:
        face_detector.close()

if __name__ == "__main__":
    main()
