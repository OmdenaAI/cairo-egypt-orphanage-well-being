import json
import cv2
import torch
import torchvision.transforms as transforms
from utils.model import Face_Emotion_CNN
from PIL import Image



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

    def __init__(self, device, model_path="project_assets/models/FER_trained_model.pt",
                 classes_path="project_assets/classes/emotion_dict.json"):
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
