from django.urls import path, re_path
from mlpipeline import views as mlpipeline_views

app_name = 'mlpipeline'

urlpatterns = [
    path('', mlpipeline_views.dashboard, name="dashboard"),
    path('cameras/', mlpipeline_views.cameras, name="cameras"),
    path('new_camera/', mlpipeline_views.new_camera, name="new_camera"),
    path('livecamera/',mlpipeline_views.livecamera, name="livecamera"),

    path('mlscript/', mlpipeline_views.mlscript, name="mlscript"),
    path('startscript/', mlpipeline_views.start_script, name="start_script"),
    path('stopscript/', mlpipeline_views.stop_script, name="stop_script"),
    ]

