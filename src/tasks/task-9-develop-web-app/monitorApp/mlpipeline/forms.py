from django import forms
from mlpipeline.models import VideoUpload

class VideoUploadForm(forms.ModelForm):
    class Meta:
        model = VideoUpload
        fields = ['title', 'video_file']
