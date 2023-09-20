from django.contrib.auth import authenticate
from django.contrib.auth.forms import (AuthenticationForm, PasswordResetForm, PasswordChangeForm,
                                        SetPasswordForm, UserCreationForm)
from django import forms
from django.contrib.auth.models import User
from django.utils.translation import gettext_lazy as _

"""For login form having username and passwords field"""
class LoginForm(AuthenticationForm):
    username = forms.CharField(required=True, label=_("Username"), max_length=254, widget=forms.TextInput(attrs={'class':'form-control inputpad','placeholder': 'username'}))
    password = forms.CharField(required=True, label=_("Password"), widget=forms.PasswordInput(attrs={'class':'form-control inputpad','placeholder': 'password'}))

    def clean(self):
        username = self.cleaned_data.get('username')
        password = self.cleaned_data.get('password')

        if username and password:
            self.user_cache = authenticate(username=username, password=password)
        return self.cleaned_data


"""For password reset form having passwords field"""
class CustomPasswordChangeForm(PasswordChangeForm):
    new_password1 = forms.CharField(
        label=_("New Password*"),
        widget=forms.PasswordInput(attrs={'placeholder': 'New Password *','class':'form-control', 'id':'newpassword1'})
    )
    new_password2 = forms.CharField(
        label=_("Confirm New Password*"),
        widget=forms.PasswordInput(attrs={'placeholder': 'Confirm New Password *','class':'form-control', 'id':'newpassword2'})
    )
    old_password = forms.CharField(
        label=_("Old Password*"),
        strip=False,
        widget=forms.PasswordInput(attrs={'placeholder': 'Old Password *','class':'form-control', 'id':'oldpassword'}),
    )

    field_order = ['old_password', 'new_password1', 'new_password2']


"""For password reset form having email field"""
class PasswordResetFormUnique(PasswordResetForm):
    email = forms.EmailField(label=_("Email"), max_length=254, widget=forms.TextInput(attrs={'class':'form-control', 'placeholder':'Email Address*', 'style':'text-transform:none;'}))

    def clean(self):
        cleaned_data = super(PasswordResetFormUnique, self).clean()
        email = cleaned_data.get('email')
        if not User.objects.filter(email=email).exists():
            raise forms.ValidationError(_("Email address not recognized. There is no account linked to this email."))
        return cleaned_data

"""Password reset form after forgot password"""
class CustomSetPasswordForm(SetPasswordForm):
    new_password1 = forms.CharField(label=_("New Password"), widget=forms.PasswordInput(attrs={'class':'form-control resetpassword ','id':'newpassword1', 'placeholder': _('New Password')}))
    new_password2 = forms.CharField(label=_("Confirm New Password"), widget=forms.PasswordInput(attrs={'class':'form-control confirm-password-reset','id':'newpassword2', 'placeholder': _('Confirm New Password')}))

"""Create User Form"""
class UserRegisterForm(UserCreationForm):
    email = forms.EmailField(required=True, label=_("Email"), max_length=254, widget=forms.TextInput(attrs={'class':'form-control', 'placeholder':'Email Address*', 'style':'text-transform:none;'}))
    username = forms.CharField(required=True, label=_("Username"), max_length=254, widget=forms.TextInput(attrs={'class':'form-control inputpad','placeholder': 'username*'}))
    password1 = forms.CharField(required=True, label=_("New Password"), widget=forms.PasswordInput(attrs={'class':'form-control ','id':'password1', 'placeholder': _('New Password*')}))
    password2 = forms.CharField(required=True, label=_("Confirm New Password"), widget=forms.PasswordInput(attrs={'class':'form-control','id':'password2', 'placeholder': _('Confirm New Password*')}))

    field_order = ['email','username','password1', 'password2']

    def clean(self):
        cleaned_data = super(UserRegisterForm, self).clean()
        email = cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError(_("Email address already linked to an account"))
        return cleaned_data
    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']  # Save the email field
        if commit:
            user.save()
        return user
