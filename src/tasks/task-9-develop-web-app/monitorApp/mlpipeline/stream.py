import cv2
import threading

class VideoCamera(object):
    def __init__(self, cameraIP):
        print(cameraIP, "inside class")
        self.video = cv2.VideoCapture(cameraIP)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        if self.frame is not None:
            _, jpeg = cv2.imencode('.jpg', self.frame)
            return jpeg.tobytes()
        else:
            return b''  # Return an empty bytes object if frame is None

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
