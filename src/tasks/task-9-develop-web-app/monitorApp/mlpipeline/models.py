# models.py
from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import pre_save
from django.dispatch import receiver
from userProfile.models import Profile

# Camera model
class Camera(models.Model):
    camera_number = models.CharField(max_length=128, verbose_name="Camera Number")
    room_details = models.CharField(max_length=128, verbose_name="Room Details")
    connected = models.BooleanField(default=False, verbose_name="Connected")
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name="cameras_created_by")
    updated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name="cameras_updated_by")
    created_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.camera_number

    class Meta:
        verbose_name_plural = "Cameras"

# Mood model
class Mood(models.Model):
    mood = models.CharField(max_length=128, verbose_name="Mood Name")
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name="moods_created_by")
    updated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name="moods_updated_by")
    created_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.mood

    class Meta:
        verbose_name_plural = "Moods"

# Activity model
class Activity(models.Model):
    activity = models.CharField(max_length=128, verbose_name="Activity Name")
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name="activities_created_by")
    updated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name="activities_updated_by")
    created_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.activity

    class Meta:
        verbose_name_plural = "Activities"

# Detection model
class Detection(models.Model):
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE, verbose_name="Profile")
    profile_name = models.CharField(max_length=128, verbose_name="Profile Name")
    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, verbose_name="Camera")
    mood = models.ForeignKey(Mood, on_delete=models.CASCADE)
    mood_name = models.CharField(max_length=128, verbose_name="Profile Name")
    activity = models.ForeignKey(Activity, on_delete=models.CASCADE)
    activity_name = models.CharField(max_length=128, verbose_name="Profile Name")
    reference_video = models.CharField(max_length=128, blank=True, null=True, verbose_name="Reference Video")
    recorded_date = models.DateTimeField()

    def __str__(self):
        return f"{self.profile_name} - {self.recorded_date}"

    class Meta:
        verbose_name_plural = "Detections"

# Signal to update profile_name, mood_name, activity_name in Detection model based on profile
@receiver(pre_save, sender=Detection)
def update_profile_name(sender, instance, **kwargs):
    instance.profile_name = instance.profile.profile_name
    instance.mood_name = instance.mood.mood
    instance.activity_name = instance.activity.activity
