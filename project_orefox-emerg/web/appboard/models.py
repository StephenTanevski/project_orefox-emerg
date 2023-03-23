from django import template
from django.db.models.fields import related
from appboard.utils.data_science.main import processor
from django.db import models
from django.contrib.auth import get_user_model
from django.core.mail import send_mail
from django.template.loader import get_template
from django.template import Context
from django.conf import settings
from django.urls import reverse

import os

UserModel = get_user_model()


"""
#################################################
Projects
#################################################
"""
class Project(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(max_length=200, unique=True)
    objective = models.CharField(max_length=250)
    locality = models.CharField(max_length=250)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.name

    def get_absolute_url(self):
        return reverse('project_details', kwargs={'project_url': self.slug})


class Member(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="members")
    user = models.ForeignKey(UserModel, on_delete=models.CASCADE, related_name="all_roles")
    is_owner = models.BooleanField(default=False)
    is_member = models.BooleanField(default=True)
    is_biller = models.BooleanField(default=False)

    # should add unique together (project, user)
    class Meta:
        unique_together = ['project', 'user']


class MemberInvitation(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="member_invitations")
    email = models.EmailField(max_length=100)
    code = models.CharField(max_length=100)
    is_owner = models.BooleanField(default=False)
    is_member = models.BooleanField(default=True)
    is_biller = models.BooleanField(default=False)
    sender = models.ForeignKey(UserModel, on_delete=models.SET_NULL, blank=True, null=True, related_name="member_invitations")

    def __str__(self) -> str:
        return f"{self.email}"

    def send(self, link=None):
        subject =  f"Invitation to join {self.project.name} Project"
        #link = f"http://{settings.SITE_HOST}/dashboard/projects/{self.project.slug}/confirmation/{self.code}/"
        template = get_template('appboard/emails/invitation_email.txt')
        context = {
            'link': link,
            'invitation': self,
        }
        message = template.render(context)
        
        if settings.DEBUG==False:
            send_mail(subject, message, settings.DEFAULT_FROM_EMAIL, [self.email,])
        else:
            print("###################################################")
            print("#################### Email ########################")
            print('')
            print(message)
            print('')
            print("#################### Email ########################")
            print("###################################################")

"""
#################################################
Projects Ends
#################################################
"""


"""
#################################################
Data Processor
#################################################
"""
def get_uploaded_file_path(instance, filename):
    return os.path.join('process_files', f"user_{instance.project.id}", 'uploaded_files', f"{filename}")
    #   "user_%d" % instance.owner.id, "car_%s" % instance.slug, filename)

def get_processed_file_path(instance, filename):
    return os.path.join('process_files', f"user_{instance.project.id}", 'processed_files', f"{filename}")


class Process(models.Model):
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name="processes")
    

    class Meta:
        verbose_name_plural = "processes"


def get_process_file_uploader_path(instance, filename):
    return os.path.join('process_files', f"proj_{instance.process.project.id}", f"{filename}")
    
class ProcessFile(models.Model):
    """
    id_1 and and id_2 is for uniquely identify a file
    id_1 standard:
        'uplaoded_file': means uploaded file/data by user
        ''
    """
    process = models.ForeignKey(Process, on_delete=models.CASCADE, related_name='process_files')
    stage_name = models.CharField(max_length=128)
    stage_action_name = models.CharField(max_length=128, null=True, blank=True)
    expected_filename = models.CharField(max_length=128, null=True, blank=True)
    other_id_1 = models.CharField(max_length=128, null=True, blank=True)
    file = models.FileField(
        upload_to=get_process_file_uploader_path,
        max_length=1000,
        null=True,
        blank=True
    )
    
"""
#################################################
Data Processor Ends
#################################################
"""