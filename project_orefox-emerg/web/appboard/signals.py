from django.db.models.signals import post_save
from django.dispatch import receiver

from . import models


@receiver(post_save, sender=models.Project)
def set_project_owner(sender, instance, created, **kwargs):
    if created:
        # now save the members table
        models.Member.objects.create(project=instance, user=instance._creator, is_owner=True, is_member=True, is_biller=True)
