from django.contrib import admin
from . import models


@admin.register(models.Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'slug', 'objective', 'locality')
    search_fields = ('name', 'slug', 'objective', 'locality')


@admin.register(models.Member)
class MemberAdmin(admin.ModelAdmin):
    list_display = ('id', 'project', 'user', 'is_owner', 'is_member', 'is_biller')
    search_fields = ('project__name', 'user__email')


@admin.register(models.MemberInvitation)
class MemberInvitationAdmin(admin.ModelAdmin):
    list_display = ('id', 'project', 'email', 'code', 'is_owner', 'is_member', 'is_biller', 'sender')


@admin.register(models.Process)
class ProcessAdmin(admin.ModelAdmin):
    list_display = ('id', 'project')

@admin.register(models.ProcessFile)
class ProcessFileAdmin(admin.ModelAdmin):
    list_display = ('id', 'process', 'file')
