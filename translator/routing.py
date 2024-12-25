from django.urls import path
from .consumers import AudioStreamConsumer

websocket_urlpatterns = [
    path('ws/', AudioStreamConsumer.as_asgi()),
]
