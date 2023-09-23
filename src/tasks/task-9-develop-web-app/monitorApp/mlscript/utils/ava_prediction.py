import torch
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,
)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
import numpy as np

contradict_actions = {
    "bend/bow (at the waist)": ["stand", "lie/sleep"],
    "crawl": ["stand", "walk", "run/jog", "jump/leap"],
    "crouch/kneel": ["stand", "lie/sleep"],
    "dance": ["lie/sleep", "sit"],
    "fall down": ["stand", "walk", "run/jog", "jump/leap"],
    "get up": ["lie/sleep", "sit"],
    "jump/leap": ["lie/sleep", "sit"],
    "lie/sleep": ["stand", "walk", "run/jog", "jump/leap", "sit"],
    "martial art": ["lie/sleep", "sit"],
    "run/jog": ["lie/sleep", "sit"],
    "sit": ["stand", "walk", "run/jog", "jump/leap"],
    "stand": ["lie/sleep", "sit"],
    "swim": ["lie/sleep", "sit"],
    "walk": ["lie/sleep", "sit"]
}


class AvaInference:
    def __init__(self, device):
        """
        Initialize the AvaInference class.

        Args:
            device (str): The device on which the model will run (e.g., 'cuda' or 'cpu').

        Attributes:
            device (str): The device on which the model will run.
            video_model (torch.nn.Module): The loaded video detection model.
            ava_labelnames (list): The label names for the video detection model.
        """
        self.device = device
        self.video_model, self.ava_labelnames = self.load_video_detection_model_and_labels()

    def ava_inference_transform(
            self, clip, boxes, num_frames=32, crop_size=640, data_mean=[0.45, 0.45, 0.45],
            data_std=[0.225, 0.225, 0.225],
            slow_fast_alpha=4
    ):
        """
        Apply AVA inference transformations to the input clip and boxes.

        Args:
            clip (Tensor): Video clip tensor of shape (T, C, H, W) where T is the number of frames,
                           C is the number of channels (3 for RGB), H is the height, and W is the width.
            boxes (List): List of bounding box coordinates, each box is represented as [x1, y1, x2, y2].
            num_frames (int): Desired number of frames in the output clip.
            crop_size (int): Target size for the shorter side when resizing frames.
            data_mean (List): Mean values for RGB normalization.
            data_std (List): Standard deviation values for RGB normalization.
            slow_fast_alpha (int or None): Alpha value for slow-fast pathway. If None, the pathway won't be used.

        Returns:
            Tuple: - Transformed clip: Depending on `slow_fast_alpha`, it can be a single tensor or a list with
            two tensors. - Transformed boxes (Tensor): Updated bounding box coordinates after transformations. -
            roi_boxes (array): Original bounding box coordinates.
        """
        # Convert boxes list to numpy array for easier operations
        boxes = np.array(boxes)

        # Make a copy of original boxes
        roi_boxes = boxes.copy()

        # Reduce the number of frames in the clip to the desired number
        clip = uniform_temporal_subsample(clip, num_frames)

        # Convert clip values from int (0-255) to float (0.0-1.0)
        clip = clip.float() / 255.0

        # Get height and width of the clip
        height, width = clip.shape[2], clip.shape[3]

        # Adjust the bounding boxes to fit within the image dimensions
        boxes = clip_boxes_to_image(boxes, height, width)

        # Scale the shorter side of each frame in the clip to `crop_size` while maintaining the aspect ratio.
        # Update the bounding boxes accordingly.
        clip, boxes = short_side_scale_with_boxes(clip, size=crop_size, boxes=boxes)

        # Normalize the clip using the given mean and standard deviation values
        clip = normalize(clip, np.array(data_mean, dtype=np.float32), np.array(data_std, dtype=np.float32))

        # Again adjust the bounding boxes to fit within the image dimensions after resizing
        boxes = clip_boxes_to_image(boxes, clip.shape[2], clip.shape[3])

        # If slow_fast_alpha is specified, create a slow and a fast pathway for the clip
        if slow_fast_alpha is not None:
            # Fast pathway uses the original clip
            fast_pathway = clip

            # Slow pathway downsamples the clip using the specified alpha value
            slow_pathway = torch.index_select(clip, 1,
                                              torch.linspace(0, clip.shape[1] - 1,
                                                             clip.shape[1] // slow_fast_alpha).long())

            # Combine the slow and fast pathways
            clip = [slow_pathway, fast_pathway]

        # Return the transformed clip, transformed bounding boxes, and original bounding boxes
        return clip, torch.from_numpy(boxes), roi_boxes

    def load_video_detection_model_and_labels(self):
        """
        Initializes and loads a pre-trained video detection model along with its label names.

        Returns:
            Tuple: A tuple containing the loaded video detection model and the corresponding label names.
        """
        # Initialize the video detection model
        video_model = slowfast_r50_detection(True).eval().to(self.device)

        # Load the label names for the video detection model
        ava_labelnames, _ = AvaLabeledVideoFramePaths.read_label_map("project_assets/classes/temp.pbtxt")

        return video_model, ava_labelnames

    def remove_contradiction(self, actions_list, contradiction_dict):
        """
        Remove contradictory actions from a list while preserving the order of actions based on probabilities.

        Args:
            actions_list (list): A list of actions sorted by probability, where higher probability actions come
            first. contradiction_dict (dict): A dictionary where keys are actions, and values are lists of actions that
            contradict them.

        Returns:
            list: A filtered list of actions without contradictory actions.
        """
        # Create a set to store actions to remove
        actions_to_remove = set()

        # Create a dictionary to store the index of each action in the list
        action_indices = {action: index for index, action in enumerate(actions_list)}

        # Iterate over the list and update actions_to_remove
        for action in actions_list:
            # Get the list of contradictory actions for the current action, default to an empty list if not found
            contradict_list = contradiction_dict.get(action, [])
            for cont_action in contradict_list:
                # Check if the contradictory action exists in the list of actions and has a lower index (higher
                # probability)
                if cont_action in action_indices and action_indices[cont_action] < action_indices[action]:
                    # Mark the current action for removal
                    actions_to_remove.add(action)
                # Check if the contradictory action exists in the list of actions and has a higher index (lower
                # probability)
                elif cont_action in action_indices and action_indices[cont_action] > action_indices[action]:
                    # Mark the contradictory action for removal
                    actions_to_remove.add(cont_action)

        # Create a new list without the conflicting actions marked for removal
        filtered_actions = [action for action in actions_list if action not in actions_to_remove]

        return filtered_actions

    def inference(self, cap, results, img_size, ):
        """
        Perform inference on a video clip.

        Args:
            cap (CustomCapture): Custom capture object containing video frames.
            results (list): List of detection results.
            img_size (int): Target size for resizing frames.


        Returns:
            dict: Updated ids_to_labels dictionary with AVA labels for each frame.
        """
        # If there are 25 frames (1 second) accumulated in the custom capture, process them
        ids_to_labels = {}
        print(f"processing {cap.idx // 25}th second clips")

        clip = cap.get_video_clip()

        if results[0].boxes[0].shape[0]:

            # Transform the video clip and its bounding boxes for inference with the AVA model
            inputs, input_boxes, _ = self.ava_inference_transform(clip, results[0].boxes[0][:, 0:4], crop_size=img_size)
            input_boxes = torch.cat([torch.zeros(input_boxes.shape[0], 1), input_boxes], dim=1)

            # Convert the inputs to a tensor and move them to the device
            if isinstance(inputs, list):
                inputs = [inp.unsqueeze(0).to(self.device) for inp in inputs]
            else:
                inputs = inputs.unsqueeze(0).to(self.device)

            # Get predictions from the slowfast model
            with torch.no_grad():
                slowfaster_preds = self.video_model(inputs, input_boxes.to(self.device))
                slowfaster_preds = slowfaster_preds.cpu()

            # Assign AVA labels to the bounding boxes based on the slowfast predictions
            for track_id, top_labels in zip(results[0].boxes[0][:, 5].tolist(),
                                            np.argsort(slowfaster_preds, axis=1)[:, -1:].tolist()):

                labels = [self.ava_labelnames[label + 1] for label in top_labels]
                labels = self.remove_contradiction(labels, contradict_actions)
                ids_to_labels[track_id] = ' '.join(labels)

            return ids_to_labels
