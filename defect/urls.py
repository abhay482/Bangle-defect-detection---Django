from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('detect/', views.detect, name='detect' )
]