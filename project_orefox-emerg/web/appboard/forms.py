from django import forms
from django.db.models import fields

from . import models

class ProjectCreationForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            field.widget.attrs['class'] = 'form-control'
    
    name = forms.CharField(max_length=100, widget=forms.TextInput(attrs={'placeholder': 'Enter Project Name', 'target': 'slug'}))
    slug = forms.CharField(max_length=200, widget=forms.TextInput(attrs={'placeholder': 'Enter Unique Project Url'}))
    objective = forms.CharField(max_length=250, widget=forms.TextInput(attrs={'placeholder': 'Enter Project Objective'}))
    locality = forms.CharField(max_length=250, widget=forms.TextInput(attrs={'placeholder': 'Enter Project Locality'}))


    class Meta:
        model = models.Project
        fields = "__all__"


class AddMemberForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            field.widget.attrs['class'] = 'form-control'
    
    name = forms.CharField(max_length=100, widget=forms.TextInput(attrs={'placeholder': 'Enter Project Name', 'target': 'slug'}))
    slug = forms.CharField(max_length=200, widget=forms.TextInput(attrs={'placeholder': 'Enter Unique Project Url'}))
    objective = forms.CharField(max_length=250, widget=forms.TextInput(attrs={'placeholder': 'Enter Project Objective'}))
    locality = forms.CharField(max_length=250, widget=forms.TextInput(attrs={'placeholder': 'Enter Project Locality'}))


    class Meta:
        model = models.Project
        fields = "__all__"
