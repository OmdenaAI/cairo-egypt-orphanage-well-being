from django.shortcuts import render, redirect
from .models import *
from .CreateEncods import CreateEncods
from .aten_monitor import aten_monitor
from django.contrib import messages

def home(request):
    if request.user_agent.is_pc:
        return render(request, 'home.html')
    else:
        return redirect('admin:index')

def app(request):
    if request.user_agent.is_pc:
        return render(request, 'app.html')
    else:
        return redirect('admin:index')

def stream(request):
    if request.user_agent.is_pc:
        profiles = orphan_list.objects.all()
        if len(profiles) > 0:
            encodings = CreateEncods()
            encodings.create_encodings(request)
            if not encodings.error_message:
                monitoring = aten_monitor()
                monitoring.start()
        else:
            messages.warning(request, 'Input at least one orphan data!')
        
        return render(request, 'stream.html')
    else:
        return redirect('admin:index')
