# models.py
from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import pre_save
from django.dispatch import receiver
from userProfile.models import Profile

# Camera model
class Camera(models.Model):
    camera_ip = models.CharField(max_length=128, verbose_name="Camera Number")
    room_details = models.CharField(max_length=128, verbose_name="Room Details")
    connected = models.BooleanField(default=False, verbose_name="Connected")
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name="cameras_created_by")
    updated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name="cameras_updated_by")
    created_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.room_details

    class Meta:
        verbose_name_plural = "Cameras"

# Detection model
class Detection(models.Model):
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE, verbose_name="Profile")
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, verbose_name="Camera")
    mood_name = models.CharField(max_length=128, verbose_name="Profile Name")
    activity_name = models.CharField(max_length=128, verbose_name="Profile Name")
    reference_video = models.CharField(max_length=128, blank=True, null=True, verbose_name="Reference Video")
    recorded_date = models.DateTimeField()

    def __str__(self):
        return f"{self.profile.profile_name} - {self.recorded_date}"

    class Meta:
        verbose_name_plural = "Detections"

class ScriptExecutions(models.Model):
    exec_start_time = models.DateTimeField(auto_now_add=True,auto_now=False)
    exec_status = models.CharField(max_length=128, verbose_name="Script Execution Status")
    exec_stop_time = models.DateTimeField(blank=True, null=True)
    exec_camera = models.ForeignKey(Camera, on_delete=models.SET_NULL, null=True, related_name="Camera")

    def __str__(self):
        return f"Script started {self.exec_start_time} is currently {self.exec_status}"

    class Meta:
        verbose_name_plural = "Script Executions"

class VideoUpload(models.Model):
    title = models.CharField(max_length=255)
    video_file = models.FileField(upload_to='videos/')  

    def __str__(self):
        return self.title

    class Meta:
        verbose_name_plural = "Uploaded Videos"