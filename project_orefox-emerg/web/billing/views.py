from django.shortcuts import redirect, render
from django.contrib.auth.decorators import login_required
import decimal
from django.contrib.auth import get_user_model

from . import models as billing_models
from appboard import models as appboard_models


UserModel = get_user_model()


@login_required
def transactions(request):
    template_name="billing/transactions.html"
    context = {}
    return render(request, template_name, context)

    
@login_required
def credits(request):
    template_name="billing/credits.html"
    context={
        'project_roles': request.user.all_roles.filter(is_owner=True),
        'user_credit': request.user.user_credit
    }
    return render(request, template_name, context)


@login_required
def user_to_project_credit(request):
    if request.method == "POST":
        data = request.POST
        transfered_credits = decimal.Decimal(data.get('credits'))
        project = appboard_models.Project.objects.get(id=int(data.get('project')))
        user_credit = request.user.user_credit

        if transfered_credits <= user_credit.credits:
            project.project_credit.credits += transfered_credits
            project.project_credit.save()
            user_credit.credits -= transfered_credits
            user_credit.save()
            # send message
        else:
            pass

    return redirect(credits)


@login_required
def project_to_user_credit(request): 
    if request.method == "POST":
        data = request.POST
        transfered_credits = decimal.Decimal(data.get('credits'))
        project = appboard_models.Project.objects.get(id=int(data.get('project')))
        project_credit = project.project_credit
        user_credit = request.user.user_credit

        if transfered_credits <= project_credit.credits:
            user_credit.credits += transfered_credits
            user_credit.save()
            project_credit.credits -= transfered_credits
            project_credit.save()
            # send message
        else:
            pass

    return redirect(credits)


@login_required
def user_to_user_credit(request): 
    if request.method == "POST":
        data = request.POST
        transfered_credits = decimal.Decimal(data.get('credits'))
        receiver = UserModel.objects.get(email=data.get('email'))
        receiver_credit = receiver.user_credit
        sender_credit = request.user.user_credit

        if transfered_credits <= sender_credit.credits:
            receiver_credit.credits += transfered_credits
            receiver_credit.save()
            sender_credit.credits -= transfered_credits
            sender_credit.save()
            # send message
        else:
            pass

    return redirect(credits)