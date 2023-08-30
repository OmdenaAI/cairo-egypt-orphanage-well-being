from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from mlpipeline.models import Camera
import cv2
from mlpipeline.stream import *
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
from urllib.parse import unquote

# Create your views here.

#Dashboard as home page
@login_required
def dashboard(request):
    return render(request, 'mlpipeline/dashboard.html')

#list all connected cameras
@login_required
def cameras(request):
    cameras = Camera.objects.all()
    if request.method == 'GET':
        return render(request, 'mlpipeline/cameras.html',{
                                                            'cameras':cameras,
            })

#create new camera
@login_required
def new_camera(request, *args, **kwargs):
    return render(request, 'mlpipeline/new_camera.html')

@gzip.gzip_page
def livecamera(request):
    try:
        encoded_url = request.GET.get('url')
        cameraIP = unquote(encoded_url)
        cam = VideoCamera(cameraIP)
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad!
        pass
