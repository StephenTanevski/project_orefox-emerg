from django.urls import path
from . import views


urlpatterns = [
    path('transactions/', views.transactions, name='transactions'),
    path('credits/', views.credits, name='credits'),
    path('credits/transfer/user-to-project/', views.user_to_project_credit, name="user_to_project_credit"),
    path('credits/transfer/project-to-user/', views.project_to_user_credit, name="project_to_user_credit"),
    path('credits/transfer/user-to-user/', views.user_to_user_credit, name="user_to_user_credit"),

]
