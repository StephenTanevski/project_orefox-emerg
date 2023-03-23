from django.urls import path
from . import views


urlpatterns = [
    path('', views.website_home, name='website_home'),
    path('message/', views.contact_message, name='contact_message'),
]
