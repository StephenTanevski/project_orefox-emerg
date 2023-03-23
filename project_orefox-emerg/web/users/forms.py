from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from allauth.account import forms as allauth_forms
from django import forms as dj_forms
from django.utils.translation import gettext as _

from .models import User


class UserCreationForm(UserCreationForm):

    class Meta(UserCreationForm):
        model = User
        fields = ('email',)


class UserChangeForm(UserChangeForm):

    class Meta:
        model = User
        fields = ('email',)


class LoginForm(allauth_forms.LoginForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            field.widget.attrs['class'] = 'form-control'
            


class SignupForm(allauth_forms.SignupForm):
    first_name = dj_forms.CharField(
        label=_("First Name"),
        widget=dj_forms.TextInput(
            attrs={"placeholder": _("First Name"), "autocomplete": "first_name"}
        )
    )
    last_name = dj_forms.CharField(
        label=_("Last Name"),
        widget=dj_forms.TextInput(
            attrs={"placeholder": _("Last Name"), "autocomplete": "last_name"}
        )
    )