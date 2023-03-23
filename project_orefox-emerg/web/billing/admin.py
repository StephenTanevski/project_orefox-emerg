from django.contrib import admin
from . import models


@admin.register(models.UserCredit)
class UserCreditAdmin(admin.ModelAdmin):
    pass


@admin.register(models.ProjectCredit)
class ProjectCreditAdmin(admin.ModelAdmin):
    pass

