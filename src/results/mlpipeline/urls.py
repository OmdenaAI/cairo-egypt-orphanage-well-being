from django.urls import path, re_path
from mlpipeline import views as mlpipeline_views

app_name = 'mlpipeline'

urlpatterns = [
    path('', mlpipeline_views.dashboard, name="dashboard"),
    path('cameras/', mlpipeline_views.cameras, name="cameras"),
    path('new_camera/', mlpipeline_views.new_camera, name="new_camera"),
    path('delete_camera/<int:camera_id>/', mlpipeline_views.delete_camera, name='delete_camera'),
    path('edit_camera/<int:camera_id>/', mlpipeline_views.edit_camera, name='edit_camera'),
    path('livecamera/',mlpipeline_views.livecamera, name="livecamera"),

    path('mlscript/', mlpipeline_views.mlscript, name="mlscript"),
    path('startscript/', mlpipeline_views.start_script, name="start_script"),
    path('stopscript/', mlpipeline_views.stop_script, name="stop_script"),
    path('stop_script_at/<int:execution_id>/', mlpipeline_views.stop_script_at, name="stop_script_at"),
    path('upload/', mlpipeline_views.video_inference_upload, name="video_inference_upload"),
    # path('view_result/<int:camera_id>/', mlpipeline_views.view_result, name="view_result"),
    # path('view_feed/<int:camera_id>/', mlpipeline_views.view_feed, name="view_feed"),
    ]

