from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from mlpipeline.models import Camera, ScriptExecutions
import cv2
from mlpipeline.stream import *
from django.views.decorators import gzip
from django.http import StreamingHttpResponse, HttpResponseRedirect
from urllib.parse import unquote
from django.urls.base import reverse_lazy
from datetime import datetime
import os
from django.conf import settings
import subprocess

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

@login_required
def mlscript(request):
    executions = ScriptExecutions.objects.all().order_by('-exec_start_time').values()
    return render(request, 'mlpipeline/mlscript.html',{'executions':executions})

@login_required
def start_script(request):
    execution = ScriptExecutions.objects.last()
    if execution.exec_status == 'Running':
        execution.exec_status = "Stop"
        execution.exec_stop_time = datetime.now()
    ScriptExecutions.objects.create(exec_status="Running")
    try:
        working_directory = os.path.join(settings.BASE_DIR, "mlscript")
        command = ["python","yolo_slowfast.py", "--input", "0", "--device", "cpu"]
        subprocess.Popen(command, cwd=working_directory)
    except Exception as e:
        print(e)
    return HttpResponseRedirect(reverse_lazy('mlpipeline:mlscript'))

@login_required
def stop_script(request):
    execution = ScriptExecutions.objects.last()
    if execution.exec_status == "Running":
        execution.exec_status = "Stop"
        execution.exec_stop_time = datetime.now()
        execution.save()
    return HttpResponseRedirect(reverse_lazy('mlpipeline:mlscript'))
