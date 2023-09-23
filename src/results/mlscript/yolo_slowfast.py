import numpy as np
import cv2
from ultralytics import YOLO
import torch
from utils.deep_sort.deep_sort import DeepSort
from utils.ava_prediction import AvaInference
from utils.custom_videocapture import MyVideoCapture
from utils.face_detection import FaceDetector
from utils.emotion_detection import EmotionDetector
from utils.person import Person, People
from utils.face_recognition import FacialRecognition
import logging
from utils.database_utils import check_script_run_status, insert_into_database


# TODO save predictions correctly in the database
# TODO Upload Video functionality
# TODO find a correct way to clear untracked predictions
# TODO Extensively test the system for any bugs


class EnhancingOrphanage:
    """
    Class for enhancing an orphanage environment through video analysis.

    Args:
        person_detector: Object detection model for detecting people.
        deepsort_tracker: DeepSORT tracker for tracking people across frames.
        face_detector: Face detection model for detecting faces.
        emotion_detector: Emotion detection model for recognizing emotions.
        face_recognizer: Facial recognition model for recognizing faces.
        action_recognizer: Action recognition model for recognizing actions.

    Attributes:
        person_detector: Object detection model for detecting people.
        deepsort_tracker: DeepSORT tracker for tracking people across frames.
        face_detector: Face detection model for detecting faces.
        emotion_detector: Emotion detection model for recognizing emotions.
        face_recognizer: Facial recognition model for recognizing faces.
        action_recognizer: Action recognition model for recognizing actions.
        embedding_list: List of face embeddings for recognition.
        name_list: List of names corresponding to face embeddings.
        people_stats: Dictionary to store information about tracked people.
        logger: Logger object for logging errors and information.

    Methods:
        detect_people(frame): Detect people in a frame and update tracking information.
        recognize_emotion_face(results, frame, gray): Recognize emotions on detected faces.
        remove_untracked_people(results): Remove untracked people from the database.
        recognize_action(cap, results): Recognize actions performed by tracked people.
        plot_one_person(person, id, frame): Plot information about a person on the frame.
        visualize_results(people, frame): Visualize information about tracked people on the frame.
        realtime_inference(source, show=True, draw_bboxes=True, save_database=False): Perform real-time video analysis.
        save_people_database(): Save the people database into a database.

    """

    def __init__(self, person_detector, deepsort_tracker, face_detector, emotion_detector,
                 face_recognizer, action_recognizer, recalculate_embedding=True):
        """
        Initialize the EnhancingOrphanage class with the provided models.

        Args:
            person_detector: Object detection model for detecting people.
            deepsort_tracker: DeepSORT tracker for tracking people across frames.
            face_detector: Face detection model for detecting faces.
            emotion_detector: Emotion detection model for recognizing emotions.
            face_recognizer: Facial recognition model for recognizing faces.
            action_recognizer: Action recognition model for recognizing actions.
        """
        self.person_detector = person_detector
        self.deepsort_tracker = deepsort_tracker
        self.face_detector = face_detector
        self.emotion_detector = emotion_detector
        self.face_recognizer = face_recognizer
        self.action_recognizer = action_recognizer
        self.embedding_list, self.name_list = face_recognizer.load_encoding(recalculate_embedding=recalculate_embedding)
        self.people_stats = People()

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create a file handler and set the logging level
        log_handler = logging.FileHandler('enhancing_orphanage.log')
        log_handler.setLevel(logging.INFO)

        # Create a logging format
        log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_handler.setFormatter(log_formatter)

        # Add the handler to the logger
        self.logger.addHandler(log_handler)

    def detect_people(self, frame):
        """
        Detect people in a frame and update tracking information.

        Args:
            frame: Input frame for object detection and tracking.

        Returns:
            results: Detection and tracking results.
        """
        try:
            results = self.person_detector(frame, verbose=False, classes=0)
            deepsort_outputs = []

            tracked_objects = self.deepsort_tracker.deepsort_update(
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
        except Exception as e:
            self.logger.error(f"Error in detect_people: {str(e)}")
            return None

    def recognize_emotion_face(self, results, frame, gray):
        """
        Recognize emotions on detected faces and update person information.

        Args:
            results: Detection and tracking results.
            frame: Input frame for emotion recognition.
            gray: Grayscale version of the input frame.
        """
        try:
            for box in results[0].boxes[0]:
                # Extract bounding box coordinates and track ID
                x1, y1, x2, y2 = box[0:4]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                track_id = box[5]
                person_frame = frame[y1:y2, x1:x2]

                # Create a new person object or retrieve an existing one based on track ID
                if int(track_id) not in self.people_stats:
                    person = Person()
                else:
                    person = self.people_stats[int(track_id)]

                # Update the person's coordinates in the frame
                person.person_coords = (x1, y1, x2, y2)
                self.people_stats[int(track_id)] = person

                # Detect the face within the person's frame
                face = self.face_detector.detect_face(person_frame, return_oneface=True, return_coordinates=True)
                if len(face) > 0:
                    x1_face, x2_face, y1_face, y2_face = face[1]

                    # Calculate the coordinates of the detected face in the original frame
                    x1_face_original = x1 + x1_face
                    y1_face_original = y1 + y1_face
                    x2_face_original = x1 + x2_face
                    y2_face_original = y1 + y2_face

                    # Resize the face region for emotion inference
                    resize_frame = cv2.resize(gray[y1_face:y2_face, x1_face:x2_face], (48, 48))

                    # Perform emotion inference on the resized grayscale face region
                    emotion = self.emotion_detector.infer_frame(resize_frame)

                    # Recognize the face and update person's name and mood
                    name = self.face_recognizer.recognize_face(face[0], self.embedding_list, self.name_list)
                    self.people_stats[track_id].name = name
                    self.people_stats[track_id].mood = emotion
                    person.face_coords = (x1_face_original, y1_face_original, x2_face_original, y2_face_original)
        except Exception as e:
            self.logger.error(f"Error in recognize_emotion_face: {str(e)}")

    def remove_untracked_people(self, results):
        """
        Remove untracked people from the statistic list.

        Args:
            results: Detection and tracking results.
        """
        try:
            # Get the set of track_ids from the results
            track_ids = set(box[5] for box in results[0].boxes[0])

            # Loop over the keys in your dict and remove the ones that are not in the track_ids set
            for key in list(self.people_stats.keys()):
                if key not in track_ids:
                    self.people_stats.pop(key, None)
        except Exception as e:
            self.logger.error(f"Error in remove_untracked_people: {str(e)}")

    def recognize_action(self, cap, results):
        """
        Recognize actions performed by tracked people.

        Args:
            cap: Video capture object.
            results: Detection and tracking results.
        """
        try:
            id_to_ava_labels = self.action_recognizer.inference(cap, results, 640)
            for trackid, action in id_to_ava_labels.items():
                self.people_stats[trackid].action = action
        except Exception as e:
            self.logger.error(f"Error in recognize_action: {str(e)}")

    def plot_one_person(self, person, id, frame):
        """
        Plot information about a person on the frame.

        Args:
            person: Person object containing information.
            id: Person's unique identifier.
            frame: Input frame for drawing information.

        Returns:
            frame: Frame with person information drawn.
        """
        if person.person_coords is None:
            return frame

        x1, y1, x2, y2 = person.person_coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), (247, 216, 0), 2)

        if person.face_coords is not None:
            x1_face, y1_face, x2_face, y2_face = person.face_coords
            cv2.rectangle(frame, (x1_face, y1_face), (x2_face, y2_face), (60, 20, 220), 2)

        # Construct the text with line breaks for readability
        line1 = f'ID: {id}, Name: {person.name}, Mood: {person.mood}, '
        line2 = f'Action: {person.action}'

        # Calculate the text size for each line
        (text_width1, text_height1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        (text_width2, text_height2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the background rectangle size
        max_text_width = max(text_width1, text_width2)
        total_text_height = text_height1 + text_height2

        # Add a semi-transparent background rectangle for text
        cv2.rectangle(frame, (x1, y1 - total_text_height - 10), (x1 + max_text_width + 10, y1), (247, 216, 0), -1)

        # Display the text on the frame
        cv2.putText(frame, line1, (x1 + 5, y1 - text_height1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, line2, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def visualize_results(self, people, frame):
        """
        Visualize information about tracked people on the frame.

        Args:
            people: Dictionary of tracked people.
            frame: Input frame for drawing information.
        """
        for id, person in people.items():
            self.plot_one_person(person, id, frame)

    def realtime_inference(self, source, show=True, draw_bboxes=True, save_database=True):
        """
        Perform real-time video analysis.

        Args:
            source: Video source (e.g., file path or camera index).
            show: Flag to display the video frame with annotations.
            draw_bboxes: Flag to draw bounding boxes around people and faces.
            save_database: Flag to save the people database into a database.
        """
        cap = MyVideoCapture(source)
        #
        while not cap.end and check_script_run_status(source) == 'Running':
            try:
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # If frame reading fails, skip the frame
                if not ret:
                    continue

                results = self.detect_people(frame)

                if results:
                    self.recognize_emotion_face(results, frame, gray)

                    if len(cap.stack) == 25:
                        self.recognize_action(cap, results)
                        if save_database:
                            self.save_people_database()

                    self.remove_untracked_people(results)

                    if draw_bboxes and show:
                        self.visualize_results(self.people_stats, frame)

                    if show:
                        cv2.imshow('Frame', frame)

                # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            except Exception as e:
                self.logger.error(f"Error in realtime_inference loop: {str(e)}")

        # When everything is done, release the video capture object
        cap.release()

        # Close all the frames
        cv2.destroyAllWindows()

    def save_people_database(self):
        """
          Save the people dictionary into the database.
        """
        insert_into_database(self.people_stats)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emotion_detector = EmotionDetector(device)
    person_detector = YOLO('project_assets/models/yolov8x')

    face_detector = FaceDetector("project_assets/models/yolov8n-face.pt")
    deepsort_tracker = DeepSort("project_assets/models/ckpt.t7", use_cuda=True)

    face_recognizer = FacialRecognition(data_path="project_assets/photos", face_detector=face_detector)

    action_recognizer = AvaInference(device)
    final = EnhancingOrphanage(person_detector, deepsort_tracker, face_detector, emotion_detector, face_recognizer,
                               action_recognizer)

    src = "project_assets/videos/Video2_chunk_2.mp4"
    final.realtime_inference(src)
