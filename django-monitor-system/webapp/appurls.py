from django.urls import path
from webapp import views

urlpatterns = [
    path('', views.app, name='app'),
    path('stream/', views.stream, name='stream'),
]
