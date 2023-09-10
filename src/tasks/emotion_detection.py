import json
import cv2
import torch
import torchvision.transforms as transforms
from model import Face_Emotion_CNN
from PIL import Image
from face_detection import detect_face

class EmotionDetector:
    """
    A class that utilizes a pre-trained CNN model to detect emotions from facial images.

    Attributes:
        device (str): The computational device used for inference (either 'cpu' or 'cuda' for GPU).
        model_path (str): Path to the pre-trained model weights.
        classes_path (str): Path to the JSON file containing the emotion class labels.
        model (torch.nn.Module): The pre-trained CNN model for emotion detection.
        val_transform (torchvision.transforms.Compose): Transformation pipeline for pre-processing input images.
    """

    def __init__(self, device, model_path, classes_path):
        """
        Initializes the EmotionDetector with the specified device and paths to the model and classes.

        Args:
            device (str): The computational device to be used ('cpu' or 'cuda').
            model_path (str): The path to the pre-trained model weights.
            classes_path (str): The path to the JSON file with the emotion class labels.
        """
        self.device = device
        self.model_path = model_path
        # Load the emotion classes from a JSON file
        with open(classes_path, 'r') as json_file:
            self.emotion_classes = json.load(json_file)
        # Load the pre-trained CNN model
        self.model = self.load_model()
        # Create an image transformation pipeline for pre-processing
        self.val_transform = self.create_transformation()

    def load_model(self):
        """
        Loads the pre-trained CNN model and sets it to evaluation mode.

        Returns:
            torch.nn.Module: The loaded pre-trained model ready for inference.
        """
        try:
            # Initialize the CNN model for facial emotion detection
            model = Face_Emotion_CNN()
            # Load the pre-trained weights into the model
            model.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
            # Move the model to the specified device (CPU or GPU)
            model.to(self.device)
            # Set the model to evaluation mode for inference
            model.eval()
            return model
        except Exception as e:
            # Catch any errors during model loading
            print(f"Error loading model: {str(e)}")
            return None

    def create_transformation(self):
        """
        Creates a transformation pipeline for pre-processing input images.

        Returns:
            torchvision.transforms.Compose: The transformation pipeline.
        """
        # Define the transformation pipeline using torchvision.transforms
        val_transform = transforms.Compose([
            transforms.ToTensor()])
        return val_transform

    def infer_frame(self, resized_gray_frame):
        """
        Performs emotion inference on a pre-processed single frame.

        Args:
            resized_gray_frame (numpy.ndarray): The input frame (image) represented as a NumPy array.

        Returns:
            str or None: The predicted emotion class label, or None if an error occurs during inference.
        """
        try:
            # Normalize the pixel values to a range [0,1]
            X = resized_gray_frame / 256
            # Convert the NumPy array to a PIL image
            X = Image.fromarray(X)
            # Apply the pre-processing transformations to the image
            X = self.val_transform(X).unsqueeze(0)
            # Disable gradient calculations as we're only doing forward pass (inference)
            with torch.no_grad():
                # Obtain the model's predictions for the input image
                log_ps = self.model(X.to(self.device))
                # Get the probabilities by applying the exponential function to the log-probabilities
                ps = torch.exp(log_ps)
                # Get the class with the highest probability
                top_p, top_class = ps.topk(1, dim=1)
                # Move the top class tensor back to CPU for NumPy operations
                top_class = top_class.cpu()
                # Get the predicted emotion label from the emotion classes dictionary
                pred = self.emotion_classes[str(int(top_class.numpy()))]
                return pred
        except Exception as e:
            # Catch any errors during inference
            print(f"Error during inference: {str(e)}")
            return None

    def realtime_inference(self, camera_source=0):
        """
        Conducts real-time emotion inference using a specified camera source.

        Args:
            camera_source (int or str): The source for the camera input, which can be an integer representing the camera index, or a string representing a video file path.
        """
        # Open a connection to the video capture source
        cap = cv2.VideoCapture(camera_source)

        # Check if the video capture source opened correctly
        if not cap.isOpened():
            print("Error: Could not open camera source.")
            return

        try:
            # Continuously capture frames from the camera source
            while True:
                # Capture a single frame
                ret, frame = cap.read()
                # Check if the frame was captured correctly
                if not ret:
                    print("Error: Failed to capture frame.")
                    break

                # Convert the frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Detect faces in the frame
                faces = detect_face(frame)
                # Loop over all detected faces in the frame
                for (x, y, w, h) in faces:
                    # Resize the face region to a fixed size (48x48 pixels)
                    resize_frame = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
                    # Draw a rectangle around the face on the original frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # Perform emotion inference on the resized grayscale face region
                    emotion = self.infer_frame(resize_frame)
                    # If an emotion was successfully predicted, display it on the frame
                    if emotion is not None:
                        cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

                # Display the frame with the emotion predictions
                cv2.imshow("Emotion Detection", frame)
                # Break the loop if the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            # Catch any errors during real-time inference
            print(f"Error during real-time inference: {str(e)}")

        # Release the video capture source and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "./models/FER_trained_model.pt"
    classes_path = "emotion_dict.json"

    detector = EmotionDetector(device, model_path, classes_path)

    # Start real-time inference from the default camera (you can specify a different source)
    detector.realtime_inference()
