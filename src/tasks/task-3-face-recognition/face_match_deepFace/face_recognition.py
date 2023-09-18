"""
    Author: Ahmed Sobhi
    Department: Data Science
    Created_at: 2023-09-10
    Objective: Inference for Face Recognition.
"""
import warnings
warnings.filterwarnings('ignore')

import recognition
import argparse

import cv2

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Inference for Face Recognition.')

# Define command-line arguments with default values
parser.add_argument('--mode', default='video', help='Handling either [image, camera, video] images in infrence (default: video)')
parser.add_argument('--img_dir', default='data/test/', help='Input directory path (default: /data/test/)')
parser.add_argument('--img_video_name', default=None, help='Input Image Name, ex: image.jpg (default: None)')
parser.add_argument('--label_path', default='data/label/', help='Label directory path (default: /data/label)')
parser.add_argument('--recognition_model', default='ArcFace', help='Recognition Model (default: ArcFace)')
parser.add_argument('--detection_model', default='retinaface', help='Detectiom Model (default: retinaface)')

# Parse the command-line arguments
args = parser.parse_args()

# Access the argument values
mode = args.mode
img_path = args.img_dir
img_video_name = args.img_video_name
label_path = args.label_path

# Defining models
face_recognition_model = args.recognition_model
face_detection_model = args.detection_model

if __name__ == '__main__':

    if mode == 'image':
        rimg = recognition.face_recognition_optimized(
            img_path=img_path+img_video_name,
            label_path=label_path,
            face_detection_model=face_detection_model,
            face_recognition_model=face_recognition_model
        )

        # Output the labeled Image
        cv2.imwrite(f"report/plots/{img_video_name}", rimg)
        
    else:
        if mode == 'camera':
            cap = cv2.VideoCapture(0)
        
        elif mode == 'video':
            cap = cv2.VideoCapture(f"{img_path}/{img_video_name}")

        # Check if the video is opened successfully
        if not cap.isOpened():
            print('Could not open the video file')
            exit()

        # Set the capture buffer size
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   

        # define counter
        i = 0
        # Loop through the video frames
        while True:
            
            ret, frame = cap.read()
            
            frame = recognition.face_recognition_optimized(
                img_path=frame,
                label_path=label_path,
                face_detection_model=face_detection_model,
                face_recognition_model=face_recognition_model
            )

            # Output the labeled Image
            cv2.imwrite(f"report/plots/{img_video_name.split('.')[0]}_{i}.jpg", frame)

            cv2.imshow('frame', frame)

            i += 1

            # Exit the loop if the user presses the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
    
    
