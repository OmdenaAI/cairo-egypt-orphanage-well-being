from django.contrib import admin

# Register your models here.
# admin.py
from django.contrib import admin
from .models import Camera, Detection

# Camera admin
class CameraAdmin(admin.ModelAdmin):
    list_display = ('camera_ip', 'room_details', 'connected')
    search_fields = ('camera_ip', 'room_details')

# Detection admin
class DetectionAdmin(admin.ModelAdmin):
    list_display = ('profile', 'profile_name', 'profile_role', 'camera', 'mood_name', 'activity_name', 'recorded_date')
    search_fields = ('profile__user__username', 'profile_name', 'camera__camera_number', 'mood_name', 'activity_name')
    readonly_fields = ('profile_name', 'profile_role',)

admin.site.register(Camera, CameraAdmin)
admin.site.register(Detection, DetectionAdmin)
