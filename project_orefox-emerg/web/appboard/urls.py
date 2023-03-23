from django.urls import path
from . import views


urlpatterns = [
    path('', views.appboard_home, name='appboard_home'),
    # path('process-file/', views.process_file, name='process_file'),
    path('process-history/', views.process_history, name='process_history'),
    
    path('new-project/', views.new_project, name='new_project'),
    path('projects/', views.project_list, name='project_list'),
    path('projects/<str:project_url>/', views.project_details, name='project_details'),
    path('projects/<str:project_url>/data-processor/', views.data_processor, name='data_processor'),
    path('projects/<str:project_url>/members/', views.project_members, name='project_members'),
    path('projects/<str:project_url>/members/manage/', views.manage_project_member, name='manage_project_member'),
    path('projects/<str:project_url>/members/invite-project-member/', views.invite_project_member, name='invite_project_member'),
    path('projects/<str:project_url>/members/project-invitation/<str:invitation_code>/', views.project_invitation, name='project_invitation'),

    path('projects/<str:project_url>/processor-file-uploader/', views.processor_file_uploader, name='processor_file_uploader'),
    path('projects/<str:project_url>/data-cleaning/', views.data_cleaning, name='data_cleaning'),
    path('projects/<str:project_url>/process-data/', views.process_data, name='process_data'),
    
    # path('projects/<str:project_url>/processor-file-upload/', views.project_details, name='processor_file_upload'),
]
