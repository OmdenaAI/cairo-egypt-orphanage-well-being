from django.urls import path
from mlpipeline import views as mlpipeline_views

app_name = 'mlpipeline'

urlpatterns = [
    path('', mlpipeline_views.dashboard, name="dashboard"),
    ]

