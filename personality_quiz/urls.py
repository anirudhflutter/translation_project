from django.urls import path
from . import views

urlpatterns = [
    path('', views.personality_quiz, name='personality_quiz'),  # For the quiz
]
