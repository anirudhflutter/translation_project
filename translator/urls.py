from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='translate'),  
    path('translate/', views.translate_view, name='translation_app'),  
    path('delete-session/<int:session_id>/', views.delete_session, name='delete_session'),
    path('fetch-dashboard-data/', views.fetch_dashboard_data, name='fetch_dashboard_data'),
    path('start-listening/', views.start_listening, name='start_listening'),
    path('stop-listening/', views.stop_listening, name='stop_listening'),
    path('generate-speech/', views.generate_speech, name='generate_speech'),
    path('is_user_enrolled/', views.is_user_enrolled, name='is_user_enrolled'),
]

