from ultralytics import YOLO
import torch
from yolo_slowfast import EnhancingOrphanage
from utils.deep_sort.deep_sort import DeepSort
from utils.ava_prediction import AvaInference
from utils.face_detection import FaceDetector
from utils.emotion_detection import EmotionDetector
from utils.face_recognition import FacialRecognition
import argparse
import os
import gdown
import zipfile





files = {
    "orphansdb.sql": "https://drive.google.com/uc?id=1wm0JRQBFPvxnoVs05thokl43xrSiB5av&export=download",
    "project_assets.zip": "https://drive.google.com/uc?id=1FQdCb7FelYbC8ABlE1C6EzwBdHEuijAl&export=download",
    "utils.zip": "https://drive.google.com/uc?id=1-BdTJkqH68jelQExwpAoLWZTzgsNNd_A&export=download"
}

def load_project_files():
    """
    Download and extract project files from specified URLs.

    Returns:
        None

    This function iterates through the files dictionary, downloads each file
    from its corresponding URL, extracts it if it's a ZIP file, and removes
    the downloaded ZIP file.

    If an error occurs during download, it will print an error message and
    continue processing the remaining files.
    """
    for filename, url in files.items():
        # Remove ".zip" extension for checking
        original_filename = filename.replace(".zip", "")

        # Check if the file already exists
        if not os.path.exists(original_filename):
            print(f"{original_filename} will be downloaded")

            try:
                # Download the file from the specified URL
                gdown.download(url, filename, quiet=False)

            except Exception as e:
                # Handle download errors
                print(f"Error downloading {filename}. Please try again.")
                print("Processing with the rest of the files.")
                continue

            # Check if the downloaded file is a .zip file
            if filename.endswith('.zip'):
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    # Extract the contents of the ZIP file
                    zip_ref.extractall()
                    print(f"Extracted contents from {filename} to {os.getcwd()}")

                # Remove the downloaded ZIP file
                os.remove(filename)
                print(f"Removed {filename}")


def main(config):
    # Check if CUDA (GPU) is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the EmotionDetector on the selected device
    emotion_detector = EmotionDetector(device)

    # Initialize the YOLO person detector
    person_detector = YOLO('project_assets/models/yolov8x')

    # Initialize the face detector using YOLO
    face_detector = FaceDetector("project_assets/models/yolov8n-face.pt")

    # Initialize the DeepSort tracker for person tracking
    deepsort_tracker = DeepSort("project_assets/models/ckpt.t7", use_cuda=True)

    # Initialize the FacialRecognition model for face recognition
    face_recognizer = FacialRecognition(data_path="project_assets/photos", face_detector=face_detector)

    # Initialize the AvaInference model for action recognition
    action_recognizer = AvaInference(device)

    # Create an instance of EnhancingOrphanage with all initialized components
    orphanage_enhancer = EnhancingOrphanage(person_detector, deepsort_tracker, face_detector, emotion_detector, face_recognizer, action_recognizer)

    orphanage_enhancer.realtime_inference(source=config.input, show=config.show, draw_bboxes=config.draw_bboxes)

if __name__ == "__main__":
        # load_project_files()
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='Real-time video analysis for enhancing an orphanage environment.')
        parser.add_argument('--input', type=str, default=0,
                            help='Video source (file path or camera index). Default: 0 (camera)')
        parser.add_argument('--show', type=bool, default=True,
                            help='Display the video feed. Default: True')
        parser.add_argument('--draw_bboxes', type=bool, default=True,
                            help='Draw bounding boxes on the frame. Default: True')      
        configuration = parser.parse_args()

        # Call the main function with the parsed configuration
        main(configuration)

