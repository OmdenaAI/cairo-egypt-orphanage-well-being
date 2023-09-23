import cv2
import torch


class MyVideoCapture:
    """
    Custom video capture class.

    This class provides methods to capture video, read frames, convert images to tensors,
    retrieve video clips, and release the video capture resources.

    Attributes:
        cap: VideoCapture object to access the video source.
        idx: Index to keep track of the frames read.
        end: Boolean flag to check if video reading has ended.
        stack: List to store frames before they are processed.
    """

    def __init__(self, source):
        """
        Initialize MyVideoCapture instance.

        Args:
            source (int or str): The video source (file path or camera index).

        Attributes initialized:
            cap: VideoCapture object to access the video source.
            idx: Starts from -1 since no frame has been read yet.
            end: Set to False initially. Will be True when reading from the source ends.
            stack: Empty list to store frames read from the source.
        """
        self.cap = cv2.VideoCapture(source)  # Create a VideoCapture object.
        self.idx = -1  # Initialize the frame index.
        self.end = False  # Initialize end flag.
        self.stack = []  # Initialize empty frame stack.

    def read(self):
        """
        Read a frame from the video capture.
        Returns:
            ret (bool): True if the frame is read correctly, otherwise False.
            img (array): Image frame read from the video. Returns None if the reading was unsuccessful.
        """
        self.idx += 1  # Increment frame index.
        ret, img = self.cap.read()  # Read a frame from the video capture.

        if ret:  # If frame read successfully,
            self.stack.append(img)  # Append frame to the stack.
        else:  # If reading was unsuccessful,
            self.end = True  # Mark the end of the video source.

        return ret, img  # Return read status and frame.

    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.

        Args:
            img (array): Image frame to be converted.

        Returns:
            tensor (Tensor): Converted image as a PyTorch tensor with an additional dimension.
        """
        # Convert image from BGR (OpenCV) format to RGB format.
        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Add a new dimension to the tensor.
        return img.unsqueeze(0)

    def get_video_clip(self):
        """
        Get a video clip as a PyTorch tensor.

        This method processes all the frames stored in the stack, converts them to tensors,
        and concatenates them to form a video clip tensor.

        Returns:
            clip (Tensor): Video clip as a PyTorch tensor.

        Raises:
            AssertionError: If there are no frames in the stack.
        """
        # Ensure that there are frames in the stack.
        assert len(self.stack) > 0, "clip length must be larger than 0!"

        # Convert each image in the stack to a tensor.
        self.stack = [self.to_tensor(img) for img in self.stack]

        # Concatenate the tensors along a new dimension and permute dimensions to get a clip.
        clip = torch.cat(self.stack).permute(-1, 0, 1, 2)

        # Clear the stack for future frames.
        del self.stack
        self.stack = []

        return clip

    # def show_frame(self):
    #     """
    #     Display the current frame.
    #
    #     This method reads and displays the current frame from the video capture.
    #     """
    #     ret, frame = self.read()  # Read a frame.
    #
    #     if ret:
    #         cv2.imshow('Video Frame', frame)  # Display the frame.
    #         cv2.waitKey(0)  # Wait for a key press.
    #     else:
    #         print("End of video source.")

    def release(self):
        """
        Release the video capture resources.

        This method should be called when you're done reading from the video source.
        """
        self.cap.release()  # Release the VideoCapture resources.
