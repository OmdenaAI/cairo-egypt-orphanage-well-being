from utils.face_detection import FaceDetector
import cv2
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1
import numpy as np

class FacialRecognition:
    def __init__(self, data_path, face_detector, device='cuda', pretrained='vggface2'):
        """
        Initialize the FacialRecognition class.

        Parameters:
        - data_path (str): Path to the dataset containing images of faces.
        - face_detector: An instance of a face detection model.
        - recognition_model: A model for face recognition.

        This constructor sets up the necessary components for face recognition.
        """
        self.face_detector = face_detector
        self.device = device
        self.pretrained = pretrained
        dataset = datasets.ImageFolder(data_path)
        self.loader = DataLoader(dataset, collate_fn=self.collate_fn)
        self.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        self.recognition_model = self.load_model()

    def collate_fn(self, x):
        """
        Collate function for DataLoader.

        Parameters:
        - x: A list of data items.

        Returns:
        - x[0]: The first item in the input list.

        This function is used as a collate function for DataLoader to retrieve the first item in a batch.
        """
        return x[0]

    def load_model(self):
        try:
            # Initialize the CNN model for facial emotion detection
            model = InceptionResnetV1(self.pretrained).eval().to(self.device)
            # Load the pre-trained weights into the model
            return model
        except Exception as e:
            # Catch any errors during model loading
            print(f"Error loading model: {str(e)}")
            return None

    def load_encoding(self, recalculate_embedding, face_images=True):
        """
        Load or recalculate face embeddings.

        Parameters:
        - recalculate_embedding (bool): If True, recalculate face embeddings; otherwise, load saved embeddings.

        Returns:
        - embedding_list (list of torch.Tensor): List of face embeddings.
        - name_list (list of str): List of corresponding names for each face.

        This function either loads precomputed face embeddings from 'data.pt' or recalculates them
        using the provided face detection and recognition models.
        """
        name_list = []
        embedding_list = []

        if recalculate_embedding:
            for img, idx in self.loader:
                if not face_images:
                    face = self.face_detector.detect_face(img, return_oneface=True)
                else:
                    face = np.array(img)
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (240, 240))

                face = self.transform(face)
                face = face.to(self.device)
                emb = self.recognition_model(face.unsqueeze(0))
                embedding_list.append(emb.detach())
                name_list.append(self.idx_to_class[idx])

            # Save data
            data = [embedding_list, name_list]
            torch.save(data, 'project_assets/classes/data.pt')  # Saving data.pt file

        load_data = torch.load('project_assets/classes/data.pt')
        embedding_list = load_data[0]
        name_list = load_data[1]

        return embedding_list, name_list

    def recognize_face(self, face, embedding_list, name_list):
        """
        Recognize a face and return the name and minimum distance.

        Parameters:
        - face (torch.Tensor): Face image tensor.
        - embedding_list (list of torch.Tensor): List of precomputed face embeddings.
        - name_list (list of str): List of corresponding names for each face.

        Returns:
        - name (str): Recognized name (or "Unknown" if not recognized).

        This function recognizes a face by comparing its embedding to the embeddings in the database.
        It returns the recognized name and the minimum distance among all database faces.
        """
        name = "Unknown"
        face = self.transform(face)
        face = face.to(self.device)
        emb = self.recognition_model(face.unsqueeze(0)).detach()
        dist_list = []

        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)

            min_dist = min(dist_list)
            min_dist_idx = dist_list.index(min_dist)
            if min_dist < 0.9:
                name = name_list[min_dist_idx]
        # print("the min_dist is", min_dist)
        return name

    def realtime_recognition(self, source, recalculate_embedding=False):
        """
        Perform real-time face recognition from a video source.

        Parameters:
        - source (str or int): Video source (e.g., file path or camera index).
        - recalculate_embedding (bool): If True, recalculate face embeddings; otherwise, load saved embeddings.

        This function performs real-time face recognition on the specified video source,
        displaying recognized faces with names and minimum distances.
        """
        cam = cv2.VideoCapture(source)
        embedding_list, name_list = self.load_encoding(recalculate_embedding=recalculate_embedding)
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab a frame, try again")
                break

            faces = self.face_detector.detect_face(frame, return_coordinates=True)

            for face, bbox in faces:
                name = self.recognize_face(face, embedding_list, name_list)

                frame = cv2.putText(frame, name + ' ', (int(bbox[0]), int(bbox[2])),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0), 1, cv2.LINE_AA)

                frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[3])), (255, 0, 0), 2)

            cv2.imshow("IMG", frame)

            k = cv2.waitKey(1)
            if k % 256 == 27:  # ESC
                print('Esc pressed, closing...')
                break

        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    detector = FaceDetector()
    recognizer = FacialRecognition(data_path="photos", face_detector=detector)

    recognizer.realtime_recognition(0)
