# Importing Required Libararies & Packages
import warnings
warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd
import cv2

from src.deepface_modified import DeepFace

# To run on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Functions Definitions
def custom_draw_on(img, faces, label_lst):
    """
        Function used to draw rectangular box around the detected faces with their labels.
    """
    dimg = img.copy()
    for i in range(len(faces)):
        face = faces[i]
        box = face['facial_area']
        color = (0, 0, 255)
        cv2.rectangle(dimg, (box['x'], box['y']), (box['x']+box['w'], box['y']+box['h']), color, 2)
        cv2.putText(dimg, label_lst[i], (box['x']-1, box['y']-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

    return dimg


def load_label_csv(label_path)->pd.DataFrame:
    """
        Function used to load csv label file
    """
    # Load Label CSV file
    # Step: Searching for a csv file locatted in this directory
    for file in os.listdir(label_path):
        if file.endswith(".csv"): 
            csv_path = os.path.join(label_path, file)

    # Step: load the csv label file
    return pd.read_csv(csv_path)


def face_recognition(
        img_path=None,
        label_path=None,
        face_detection_model='retinaface',
        face_recognition_model='ArcFace',
        enforce_detection=True,
        ):
    """
        This function could be used using the original deepface package without any modification in the package.
    """
    # Load Label CSV file
    df_label = load_label_csv(label_path=label_path)

    # Detect Faces
    detected_faces = DeepFace.extract_faces(
        img_path=img_path,
        detector_backend=face_detection_model
    )

    # Recognitize Faces with labels in database
    found_faces = DeepFace.find(img_path=img_path,
                db_path=label_path,
                model_name=face_recognition_model,
                detector_backend=face_detection_model,
                enforce_detection=enforce_detection)

    # Define label list
    label_lst = []

    for found_face in found_faces:
        
        img_label = 'UnKown'

        if len(found_face) != 0:

            # Face was detected
            # Pick the highest similarity
            found_face = found_face[
                found_face.ArcFace_cosine == found_face.ArcFace_cosine.max()
                ]
            
            # checking the ArcFace_cosine threshold
            thr = 0.25
            
            if found_face['ArcFace_cosine'].values[0]> thr:
                img_label_path = found_face['identity'].values[0]
                img_label = img_label_path.split('/')[-1].split('.')[0]
                # img_label = df_label[df_label.image_name == img_label_path[img_label_path.rfind('/') + 1:]]['label'].values[0]
                
        label_lst.append(img_label)

        
    if isinstance(img_path, str):
        img = cv2.imread(img_path)

        rimg= custom_draw_on(img, detected_faces, label_lst)

        print(f"Recognitioned Faces in {img_path} are: {label_lst}. Exported image with detected faces at: ../report/plots/{img_path.split('/')[-1]}")

    else:
        rimg= custom_draw_on(img_path, detected_faces, label_lst)

    return rimg


def face_recognition_optimized(
        img_path=None,
        label_path=None,
        face_detection_model='retinaface',
        face_recognition_model='ArcFace',
        enforce_detection=True,
        ):
    """
        This function could only be used by using the modified deepface package.
    """
    # Load Label CSV file
    df_label = load_label_csv(label_path=label_path)

    # Detect Faces
    detected_faces = DeepFace.extract_faces(
        img_path=img_path,
        detector_backend=face_detection_model
    )

    found_faces = DeepFace.find_modified(detected_faces,
                db_path=label_path,
                model_name=face_recognition_model,
                detector_backend='skip',
                enforce_detection=enforce_detection)

    # Define label list
    label_lst = []

    for found_face in found_faces:
        
        img_label = 'UnKown'

        if len(found_face) != 0:

            # Face was detected
            # Pick the highest similarity
            found_face = found_face[
                found_face.ArcFace_cosine == found_face.ArcFace_cosine.max()
                ]
            
            # checking the ArcFace_cosine threshold
            thr = 0.25
            
            if found_face['ArcFace_cosine'].values[0]> thr:
                img_label_path = found_face['identity'].values[0]
                img_label = img_label_path.split('/')[-1].split('.')[0]
                # img_label = df_label[df_label.image_name == img_label_path[img_label_path.rfind('/') + 1:]]['label'].values[0]
                
        label_lst.append(img_label)

    if isinstance(img_path, str):
        img = cv2.imread(img_path)

        rimg= custom_draw_on(img, detected_faces, label_lst)

        print(f"Recognitioned Faces in {img_path} are: {label_lst}. Exported image with detected faces at: ../report/plots/{img_path.split('/')[-1]}")

    else:
        rimg= custom_draw_on(img_path, detected_faces, label_lst)

    return rimg