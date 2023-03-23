from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
from django.contrib.auth import get_user_model


from appboard.models import Project
from . import models


UserModel = get_user_model()


@receiver(post_save, sender=UserModel)
def create_user_credit(sender, instance, created, **kwargs):
    if created:
        models.UserCredit.objects.create(user=instance)


@receiver(post_save, sender=Project)
def create_project_credit(sender, instance, created, **kwargs):
    if created:
        models.ProjectCredit.objects.create(project=instance)


