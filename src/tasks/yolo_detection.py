import torch
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert tensor to PIL Image to apply transforms
    transforms.Resize((240, 240)),  # Resize or you can use `CenterCrop` if you wish to crop
    transforms.ToTensor()  # Convert back to tensor
])

def detect_face(face_detector, img):
    """
    Detect faces in an image using a face detection model.

    Args:
        face_detector (YOLO): The face detection model to use.
        img (numpy.ndarray): The input image containing faces to be detected.

    Returns:
        list: A list of tuples. Each tuple contains:
            - The detected face image.
            - Bounding box coordinates [x, y, width, height] of the detected face.
            - Confidence score of the detection.
    """
    # Initialize an empty list to store the response for each detected face
    faces_info = []

    # Compute the scaling factors
    scale_factor_x = img.shape[1] / 240.0
    scale_factor_y = img.shape[0] / 240.0

    # Detect faces using the face detection model
    results = face_detector.predict(img, verbose=False, show=False, conf=0.25)[0]

    # Iterate over each detected face
    for result in results:
        # Extract the bounding box coordinates and confidence score
        x, y, w, h = result.boxes.xywh.tolist()[0]
        confidence = result.boxes.conf.tolist()[0]

        # Adjust the bounding box coordinates
        x, y, w, h = int(x - w / 2), int(y - h / 2), int(w), int(h)

        # Adjust bounding boxes to match the original image size
        x = int(x * scale_factor_x)
        y = int(y * scale_factor_y)
        w = int(w * scale_factor_x)
        h = int(h * scale_factor_y)

        # Crop the detected face region from the original image
        detected_face = img[y: y + h, x: x + w].copy()
        detected_face = torch.from_numpy(detected_face.transpose(2, 0, 1))  # Convert HWC to CHW
        detected_face = transform(detected_face)

        # Append the detected face, bounding box coordinates, and confidence to the response list
        faces_info.append((detected_face, [x, y, w, h], confidence))

    # Return the list of responses containing detected face information
    return faces_info
