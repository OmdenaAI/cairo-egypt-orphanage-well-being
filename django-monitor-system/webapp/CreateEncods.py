from django.contrib import messages
from fr_as.settings import MEDIA_ROOT
from .models import *
import cv2
import face_recognition
import urllib
import numpy as np
import pickle
import base64


class CreateEncods:
    def __init__(self):
        self.error_message = []
        
    def create_encodings(self, request):
        local_host = 'http://127.0.0.1:8000'
        objects = orphan_list.objects.all()
        if len(objects) > 0:
            for obj in objects:
                if obj.photo1:
                    if not obj.encoding1:
                        photo = f'{obj.name}, Photo 1'
                        encod = self.img_to_encod(request, obj.photo1, photo, local_host+obj.photo1.url)
                        obj.encoding1 = self.convert(encod)
                else:
                    obj.encoding1 = b''

                if obj.photo2:
                    if not obj.encoding2:
                        photo = f'{obj.name}, Photo 2'
                        encod = self.img_to_encod(request, obj.photo2, photo, local_host+obj.photo2.url)
                        obj.encoding2 = self.convert(encod)
                else:
                    obj.encoding2 = b''

                if obj.photo3:
                    if not obj.encoding3:
                        photo = f'{obj.name}, Photo 3'
                        encod = self.img_to_encod(request, obj.photo3, photo, local_host+obj.photo3.url)
                        obj.encoding3 = self.convert(encod)
                else:
                    obj.encoding3 = b''

                obj.save()
            
            self.message_user(request)

    def img_to_encod(self, request, path, photo, url):
        resp = urllib.request.urlopen(url)
        arr = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        resized_img = imutils.resize(img, width=500)

        filepath = os.path.join(MEDIA_ROOT, str(path))
        cv2.imwrite(filepath, resized_img)

        encode = np.array([])
        faceEncodList = face_recognition.face_encodings(resized_img)

        if len(faceEncodList) == 0:
            messages.error(request, f'Cannot Find a Face in {photo}')
            self.error_message.append(f'Cannot Find a Face in {photo}')
        elif len(faceEncodList) > 1:
            messages.error(request, f'Multiple Faces in {photo}')
            self.error_message.append(f'Multiple Faces in {photo}')
        else:
            encode = faceEncodList[0]

        return encode

    def convert(self, encod):
        if encod.size > 0:
            np_bytes = pickle.dumps(encod)
            np_base64 = base64.b64encode(np_bytes)
            return np_base64

    def message_user(self, request):
        if not self.error_message:
            messages.success(request, 'Face Encodings have been successfully generated!')
