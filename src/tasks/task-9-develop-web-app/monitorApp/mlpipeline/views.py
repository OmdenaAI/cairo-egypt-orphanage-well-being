from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from mlpipeline.models import Camera, ScriptExecutions, Detection, VideoUpload
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
from mlpipeline.forms import VideoUploadForm
from django.contrib import messages
import numpy as np

import sys


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
        print(camera.camera_ip,"camera_ip")
        print(camera.id,"camera_iD")
        execution = ScriptExecutions.objects.filter(exec_camera=camera).last()
        if execution:
            if execution.exec_status == 'Running':
                execution.exec_status = "Stop"
                execution.exec_stop_time = time_now
        ScriptExecutions.objects.create(exec_status="Running", exec_camera = camera)
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = r'D:\projects\cairo-egypt-orphanage-well-being\src\tasks\task-9-develop-web-app\.env\Lib\site-packages'
            working_directory = os.path.join(settings.BASE_DIR, "mlscript")
            python_executable = sys.executable
            command = [python_executable,"main.py", "--input", camera.camera_ip]
            subprocess.Popen(command, cwd=working_directory, env=env)
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

@login_required
def video_inference_upload(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()  # Save the uploaded video to the database
            camera = Camera.objects.create(  camera_ip = os.path.join(settings.BASE_DIR, "media", "videos", str(form.cleaned_data['video_file'])),
                                    room_details = form.cleaned_data['title'],
                                    connected = False
                                    )
            ScriptExecutions.objects.create(exec_status="Running", exec_camera = camera)         
            return redirect('mlpipeline:view_feed', camera_id=camera.id)
    else:
        form = VideoUploadForm()

    return render(request, 'mlpipeline/video_inference.html', {'form': form})

# Decorator to enable Gzip compression for the response
@gzip.gzip_page
def view_result(request, camera_id, *args, **kwargs):
    # Function to generate video frames from your script
    camera = get_object_or_404(Camera, pk=camera_id)
    def generate_frames(camera):
        # Replace this path with the actual path to your script
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = r'D:\projects\cairo-egypt-orphanage-well-being\src\tasks\task-9-develop-web-app\.env\Lib\site-packages'
            working_directory = os.path.join(settings.BASE_DIR, "mlscript")
            python_executable = sys.executable
            command = [python_executable,"main.py", "--input", camera.camera_ip, "--web_app", "True"]
            process = subprocess.Popen(command, cwd=working_directory, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            while True:
                stdout_line = process.stdout.readline()
                if stdout_line:
                    print("stdout:", stdout_line.decode().strip())  # Decode from bytes to string

                # Read a line from stderr
                stderr_line = process.stderr.readline()
                if stderr_line:
                    print("stderr:", stderr_line.decode().strip())  # Decode from bytes to string

                # output, errors = process.communicate()
                output = process.stdout.read(1024) 
                
                # Extract and yield frames from the script output
                try:
                    frame = cv2.imdecode(np.frombuffer(output, dtype=np.uint8), cv2.IMREAD_COLOR)
                except:
                    continue
                if frame is not None:
                    _, buffer = cv2.imencode('.jpg', frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            print("Django Error:",e)
        finally:
            if process and process.poll() is None:
                process.terminate()

        # Start the script as a subprocess

    # Clean up the subprocess when the view is done
    response = StreamingHttpResponse(generate_frames(camera), content_type="multipart/x-mixed-replace;boundary=frame")
    return response

def view_feed(request, camera_id):
    camera = get_object_or_404(Camera, pk=camera_id)
    return render(request, 'mlpipeline/view_feed.html', {'camera':camera}) 