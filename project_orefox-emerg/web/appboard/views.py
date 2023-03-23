import string
import random
from django import template
from django.shortcuts import render, redirect
from django.urls import reverse
from django.contrib import messages
from django.views import View
from django.core.files.base import ContentFile
from django.utils import timezone
import datetime
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.core import serializers
import json
import os



from . import models
from . import forms
from .utils.data_science.main import processor
from data_science import utils as ds_utils
from data_science import cleaner
from data_science.main import easy_processor

def generate_random_string(size=10, chars=string.ascii_lowercase+string.digits):
    return ''.join(random.choice(chars) for _ in range(size))



@login_required
def appboard_home(request):
    template_name = 'appboard/home.html'
    context = {
        'processes': models.Process.objects.all(),
    }
    return render(request, template_name, context=context)

@login_required
def process_file_upload(request):
    form_data = request.POST
    uploaded_file = request.FILES['file_input']
    if request.method == "POST" and uploaded_file:
        # process_files = models.ProcessFile.objects.filter(user=request.user).filter(paid=False)
        # if len(process_files) < 15:
        #     pass
        pfile = models.ProcessFile.objects.create(
            user = request.user,
            uploaded_file = uploaded_file,
            process_name = form_data['process_name']
        )

        # should be integrate within other method or should be asynch with celery
        file_path = pfile.uploaded_file.path
        output = processor(process=pfile.process_name, file_path=file_path)
        content = output['content']
        file_name = output['file_name']
        pfile.processed_file.save(
            file_name, ContentFile(content.getvalue())
        )
        pfile.save()


        context = {}
        return redirect('process_history')



# def process_file(request):
#     form_data = request.POST
#     if request.method == "POST":
#         for key, value in form_data.items():
#             if key[:6] == "check_" and value=="on": 
#                 id = int(key[6:])
#                 pfile = models.ProcessFile.objects.get(id=id)
#                 if pfile.paid and not pfile.processed_file:
#                     file_path = pfile.uploaded_file.path
#                     output = processor(process=pfile.process_name, file_path=file_path)
#                     content = output['content']
#                     file_name = output['file_name']
#                     pfile.processed_file.save(
#                         file_name, ContentFile(content.getvalue())
#                     )
#                     pfile.save()

#         return redirect('data_processor')

@login_required
def process_history(request):
    template_name = 'appboard/process_history.html'
    filter_days = 7
    context = {
        'filter_days': 7,
        'process_files': models.ProcessFile.objects.filter(user=request.user, upload_time__gte=timezone.now()-datetime.timedelta(days=filter_days)),
    }
    return render(request, template_name, context)



@login_required
def new_project(request):
    template_name = 'appboard/new_project.html'
    form =  forms.ProjectCreationForm()
    context = {
        'form': form,
        'members': models.Member.objects.filter(user=request.user),
    }
    if request.method == "POST":
        form = forms.ProjectCreationForm(request.POST)
        if form.is_valid():
            project = form.save(commit=False)
            # manipulate data here in the same form
            project._creator = request.user
            project.save()
            messages.success(request, f'"{project.name}" Project has been created!')

    return render(request, template_name, context)


@login_required
def project_list(request):
    template_name =  'appboard/project_list.html'
    context = {
        'members': models.Member.objects.filter(user=request.user),
    }
    return render(request, template_name=template_name, context=context)


@login_required
def project_details(request, project_url):
    template_name = 'appboard/project_details.html'
    project = models.Project.objects.get(slug=project_url)
    context = {
        'project': project,
        # 'members': models.Member.objects.filter(user=request.user),
        'projcet_credits': project.project_credit.credits,
    }
    return render(request, template_name=template_name, context=context)


@login_required
def project_members(request, project_url):
    template_name = "appboard/project_members.html"
    project = models.Project.objects.get(slug=project_url)
    member = project.members.get(user=request.user)
    context = {
        'project': project,
        'project_url': project_url,
        'is_owner': member.is_owner,
    }
    return render(request, template_name, context)

@login_required
def manage_project_member(request, project_url):
    if request.method == "POST":
        data = request.POST
        project_member = models.Member.objects.get(project__slug=project_url, user__email=data.get("target_member_email"))
        if data.get("delete_member"):
            if data.get("target_member_email")==request.user.email:
                messages.warning(request, f"""<b>Warning!</b> you can not delete yourself. Either another <b>Owner</b> needs to remove you, or delete the whole project. <br>
                <small>Deleting project will remove all the members automatically.</small>""")
            else:
                project_member.delete()
                messages.success(request, f"""<b>Success!</b> member <b>'{data.get("target_member_email")}'</b> has been deleted.""")
        
        elif data.get("update_member"):
            project_member.is_owner=True if data.get("is_owner") else False
            project_member.is_biller=True if data.get("is_biller") else False
            project_member.save()
            messages.success(request, f"""<b>Success!</b> member <b>'{data.get("target_member_email")}'</b> roles have been updated.""")


    return redirect("project_members", project_url=project_url)

@login_required
def invite_project_member(request, project_url):
    if request.method == "POST":
        data = request.POST
        invitation_code = generate_random_string(size=15, chars=string.ascii_uppercase+string.digits)
        member_invitation = models.MemberInvitation.objects.create(
            project = models.Project.objects.get(slug=project_url),
            email = data['email'],
            code = invitation_code,
            is_owner = True if data.get('is_owner') else False, 
            is_member= True if data.get('is_member') else False, 
            is_biller= True if data.get('is_biller') else False,
            sender = request.user
        )
        link = reverse("project_invitation", kwargs={"project_url": project_url,"invitation_code": invitation_code})
        link = request.build_absolute_uri(link)
        member_invitation.send(link=link)

        messages.success(request, f"Success! An invitation E-mail has sent to {data['email']}")

    return redirect(project_members, project_url=project_url)

@login_required
def project_invitation(request, project_url, invitation_code):
    if request.method == "POST":
        data = request.POST
        member_invitation = models.MemberInvitation.objects.get(project__slug=project_url, code=invitation_code)
        if data.get('accept_invitation'):
            if request.user.email == member_invitation.email:
                models.Member.objects.create(
                    project = member_invitation.project,
                    user = request.user,
                    is_owner = member_invitation.is_owner,
                    is_member = member_invitation.is_member,
                    is_biller = member_invitation.is_biller
                )
                messages.success(request, f"Success! You've accepted invitation")
            else:
                messages.error(request, f"Error! You are not allowed to do this operation")
            
            return redirect('appboard_home')

        elif data.get('delete_invitation'):
            if request.user.email == member_invitation.email:
                member_invitation.delete()
                messages.success(request, f"Success! You've deleted invitation")
            else:
                messages.error(request, f"Error! You are not allowed to do this operation")

            return redirect('appboard_home')
    
    template_name = "appboard/project_invitation.html"
    context = {
        "project_url": project_url,
        "invitation_code": invitation_code,
    }
    return render(request, template_name=template_name, context=context)

@login_required
def data_processor(request, project_url):
    """
    Should set permission later
    """
    template_name = "appboard/data_processor.html"
    project = models.Project.objects.get(slug=project_url)
    context = {
        "project": project,
        "project_credits": project.project_credit.credits,
        "project_url": project_url,
        "recommended_libraries": None,
    }
    return render(request, template_name, context=context)

@login_required
def processor_file_uploader(request, project_url):
    if request.method == 'POST':
        data = request.POST
        uploaded_file = request.FILES.get('uploaded_file')
        process = models.Process.objects.create(project=models.Project.objects.get(slug=project_url))
        process_file = models.ProcessFile.objects.create(process=process, stage_name="load_data", file=uploaded_file)
        filepath = process_file.file.path
        filename = os.path.basename(filepath)
        is_csv = True if filename.split('.')[-1]=="csv" else False
        is_excel = True if filename.split('.')[-1]=="xlsx" else False
        is_xls = True if filename.split('.')[-1]=="xls" else False

        df = ds_utils.get_data_frame(filepath, is_csv, is_xls, is_excel)
        missing_percentage = ds_utils.get_missing_percentage(df)

        sheet_names = None
        if is_xls or is_excel:
            sheet_names = ds_utils.get_sheet_names(filepath)

        data = {
            "file_uploaded": True,
            "process_id": process.id,
            'sheet_names': sheet_names,
            "missing_percentage": missing_percentage
        }
        return JsonResponse(data, content_type="application/json")


@login_required
def data_cleaning(request, project_url):
    if request.method == "POST":
        data = request.POST
        process_id = int(data.get('process_id'))
        lab_name = data.get('lab')
        process = models.Process.objects.get(id=process_id)

        dc = cleaner.DataCleaner(process.uploaded_file.path, lab=lab_name)
        cleaner_content = dc.write_csv("output.csv")
        process.processed_file.save(
            "output.csv", ContentFile(cleaner_content.getvalue())
        )

        data = {
            'data_cleaned': True,
        }
        return JsonResponse(data, content_type="application/json")


@login_required
def process_data(request, project_url):
    if request.method == "POST":
        form_data = request.POST
        index_col = int(form_data.get('index_col')) if form_data.get('index_col') else None
        sheet_name = int(form_data.get('sheet_name')) if form_data.get('sheet_name') else None
        process_obj = models.Process.objects.get(id = int(form_data.get('process_id')))
        cleaner_lab = form_data.get('cleaner_lab') if form_data.get('cleaner_lab') else None
        cleaner_unit = form_data.get('cleaner_unit') if form_data.get('cleaner_unit') else None
        cleaner_action = form_data.get('cleaner_action') if form_data.get('cleaner_action') else None
        cleaner_imputing_action = form_data.get('cleaner_imputing_action') if form_data.get('cleaner_imputing_action') else None
        analyser_action = form_data.get('analyser_action') if form_data.get('analyser_action') else None
        plotter_action = form_data.get('plotter_action') if form_data.get('plotter_action') else None
        report_action = form_data.get('report_action') if form_data.get('report_action') else None

        success = False
        try:
            success = True
            easy_processor(
                process_obj = process_obj,
                index_col = index_col, 
                sheet_name = sheet_name,
                cleaner_lab = cleaner_lab,
                cleaner_unit = cleaner_unit,
                cleaner_action = cleaner_action,
                cleaner_imputing_action = cleaner_imputing_action,
                analyser_action = analyser_action,
                plotter_action = plotter_action,
                report_action = report_action)
        except:
            success = False
        
        if success:
            data = {
                'success': success,
                'analysis_report': process_obj.process_files.filter(stage_name="report", stage_action_name="analysis_report").first().file.url
            }
        else:
            data = {
                'success': success
            }
        return JsonResponse(data, content_type="application/json")
