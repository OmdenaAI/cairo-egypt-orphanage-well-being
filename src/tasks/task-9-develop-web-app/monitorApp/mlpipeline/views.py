from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from mlpipeline.models import Camera, ScriptExecutions, Detection
from userProfile.models import orphanageRoles
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
from django.db.models import Count, Max, F, Subquery, OuterRef, DateTimeField

from django.contrib import messages

# Create your views here.

#Dashboard as home page
@login_required
def dashboard(request):
    search_data={}
    cameras = Camera.objects.filter(connected=True).values('id','room_details')
    roles = orphanageRoles.objects.all()
    if request.GET.get('camera'):
        search_data['camera_id'] = request.GET.get('camera')
    if request.GET.get('role'):
        search_data['profile__role_id'] = request.GET.get('role')
    camera_id = int(search_data['camera_id']) if 'camera_id' in search_data.keys() else None
    role_id = int(search_data['profile__role_id']) if 'profile__role_id' in search_data.keys() else None
    data_type = request.GET.get('data_type')
    if request.GET.get('data_type') == '0':
        mood_detections = Detection.objects.filter(**search_data).values('mood_name').annotate(counts=Count('mood_name'))
        activity_detections = Detection.objects.filter(**search_data).values('activity_name').annotate(counts=Count('activity_name'))
    else:
        # Create a subquery to find the latest recorded_date for each camera
        latest_recorded_dates = Detection.objects.filter(
            camera_id=OuterRef('camera_id')
        ).order_by('-recorded_date').values('recorded_date')[:1]

        # Use the subquery in the main query to retrieve the mood vs. count for the last recorded date for each specific camera
        mood_detections = Detection.objects.filter(**search_data,recorded_date=Subquery(latest_recorded_dates),
                    ).values('mood_name').annotate(counts=Count('mood_name')).order_by('mood_name')

        activity_detections = Detection.objects.filter(**search_data,recorded_date=Subquery(latest_recorded_dates),
                    ).values('activity_name').annotate(counts=Count('activity_name')).order_by('activity_name')
    return render(request, 'mlpipeline/dashboard.html', {'mood_detections':mood_detections,'activity_detections':activity_detections,
                                                                'cameras':cameras,'roles':roles,
                                                                'camera_id':camera_id,'role_id':role_id,'data_type':data_type})

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
    if request.method == 'POST':
        Camera.objects.create(camera_ip = request.POST.get('camera_ip'),
                            room_details = request.POST.get('room_details'),
                            connected = request.POST.get('connected'))
        return HttpResponseRedirect(reverse_lazy('mlpipeline:cameras'))
    return render(request, 'mlpipeline/new_camera.html')

#edit camera
@login_required
def edit_camera(request, camera_id, *args, **kwargs):
    if request.method == 'POST':
        Camera.objects.filter(id=camera_id).update(camera_ip = request.POST.get('camera_ip'),
                            room_details = request.POST.get('room_details'),
                            connected = request.POST.get('connected'))
        return HttpResponseRedirect(reverse_lazy('mlpipeline:cameras'))
    else:
        camera = get_object_or_404(Camera, pk=camera_id)
        return render(request, 'mlpipeline/new_camera.html',{'camera':camera})

"""Delete Camera - Starts"""
@login_required
def delete_camera(request, camera_id):
    camera = get_object_or_404(Camera, pk=camera_id)
    if request.user.is_superuser:
        camera.delete()
        messages.success(request, 'Camera deleted successfully.')
    return HttpResponseRedirect(reverse_lazy('mlpipeline:cameras'))
"""Delete Camera - Ends"""

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
    cameras = Camera.objects.filter(connected=True)
    time_now = datetime.now()
    for camera in cameras:
        execution = ScriptExecutions.objects.filter(exec_camera=camera).last()
        if execution:
            if execution.exec_status == 'Running':
                execution.exec_status = "Stop"
                execution.exec_stop_time = time_now
        ScriptExecutions.objects.create(exec_status="Running", exec_camera = camera)
        try:
            working_directory = os.path.join(settings.BASE_DIR, "mlscript")
            command = ["python","yolo_slowfast.py", "--input", camera.camera_ip, "--output", f"output_{camera.id}_{time_now}.mp4", "--device", "cpu"]
            subprocess.Popen(command, cwd=working_directory)
        except Exception as e:
            print(e)
    return HttpResponseRedirect(reverse_lazy('mlpipeline:mlscript'))

@login_required
def stop_script(request):
    ScriptExecutions.objects.filter(exec_status = "Running").update(exec_status = "Stop", exec_stop_time = datetime.now())
    return HttpResponseRedirect(reverse_lazy('mlpipeline:mlscript'))

@login_required
def stop_script_at(request, execution_id):
    ScriptExecutions.objects.filter(id = execution_id).update(exec_status = "Stop", exec_stop_time = datetime.now())
    return HttpResponseRedirect(reverse_lazy('mlpipeline:mlscript'))
