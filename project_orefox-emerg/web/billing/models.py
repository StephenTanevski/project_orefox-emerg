from django.db import models
from django.contrib.auth import get_user_model

from appboard.models import Project


UserModel = get_user_model()


class UserCredit(models.Model):
    user = models.OneToOneField(UserModel, on_delete=models.CASCADE, related_name="user_credit")
    credits = models.DecimalField(max_digits=10, decimal_places=2, default=0)

    def __str__(self) -> str:
        return f"{self.credits}"


class ProjectCredit(models.Model):
    project = models.OneToOneField(Project, on_delete=models.CASCADE, related_name="project_credit")
    credits = models.DecimalField(max_digits=10, decimal_places=2, default=0)

    def __str__(self) -> str:
        return f"{self.credits}"
