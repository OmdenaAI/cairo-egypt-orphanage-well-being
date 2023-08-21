from django.shortcuts import render, get_object_or_404
from django.conf import settings
from django.contrib.auth import login, logout, update_session_auth_hash
from userProfile.forms import LoginForm, CustomPasswordChangeForm, PasswordResetFormUnique, UserRegisterForm
from userProfile.custom_authentication import EmailAuthBackend
from userProfile.models import orphanageRoles, Profile
from django.http import HttpResponseRedirect
from django.urls.base import reverse_lazy
from django.contrib.auth.decorators import login_required
from django.utils.translation import gettext_lazy as _
from django.contrib.auth import views as auth_views
from django.contrib.auth.models import User

from django.contrib import messages

# Create your views here.

"""Sign Up View - starts"""
def signup_view(request):
    if request.user.is_authenticated:
        logout(request)
    if request.method == 'GET':
        signup_form = UserRegisterForm()
        return render(request, 'userProfile/signup.html', {'signup_form':signup_form})
    elif request.method == 'POST':
        signup_form = UserRegisterForm(request.POST)
        if signup_form.is_valid():
            user = signup_form.save()
            message = _("Your account has been created! Please log in using your credentials.")
            return HttpResponseRedirect(reverse_lazy('userProfile:login'))
        else:
            message = _('There was an error in the registration form. Please correct the errors.')
        return render(request, 'userProfile/signup.html', { 'signup_form': signup_form,
                                                            'message':message
                                                            })
"""Sign Up View - ends"""

"""Login Functionality - starts"""
def login_view(request):
    AUTH_BACKEND = settings.AUTHENTICATION_BACKENDS[0]
    if request.method == 'GET':
        if request.user.is_authenticated:
            return HttpResponseRedirect(reverse_lazy('userProfile:logout'))
        else:
            form = LoginForm(auto_id=False)
        return render(request, 'userProfile/login.html',{'form':form,})

    if request.method == 'POST':
        form = LoginForm(request, data=request.POST, auto_id=False)
        next = request.POST.get('next')
        if form.is_valid():
            obj_auth = EmailAuthBackend()
            user = obj_auth.authenticate(username=request.POST.get('username'),password=request.POST.get('password') , request=request)
            if user is not None:
                login(request, user, backend=AUTH_BACKEND)
                # if user.is_superuser and user.is_staff:
                #     return HttpResponseRedirect(reverse_lazy('admin:index'))
                # else:
                if not Profile.objects.filter(user_id=user.id).exists():
                    return HttpResponseRedirect(reverse_lazy('userProfile:profile'))
                elif next:
                    return HttpResponseRedirect(next)
                else:
                    return HttpResponseRedirect(reverse_lazy('mlpipeline:dashboard'))
            else:
                message = _("Invalid User ID or Password")
        else:
            message = _('Login failed! please verify the User ID and Password')
        return render(request, 'userProfile/login.html',{'form':form,
                                                    'message':message,})
"""Login Functionality - ends"""

"""logout functionality. - Starts"""
def logout_view(request):
    logout(request)
    return HttpResponseRedirect(reverse_lazy('userProfile:login'))
"""logout functionality. - Ends"""

"""Change Password functionality. - Starts"""
@login_required
def change_password(request):
    if request.method == 'POST':
        form = CustomPasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)
            request.session['succes_msg'] = str(_('Password has been changed successfully'))
            return HttpResponseRedirect(reverse_lazy('userProfile:logout'))
        else:
            message = _("Invalid Password")
            return render(request, 'userProfile/change_password.html', {  'form': form,
                                                                        'message':message
                                                                        })
    else:
        form = CustomPasswordChangeForm(request.user)
    return render(request, 'userProfile/change_password.html', {'form': form })
"""Change Password functionality. - Ends"""

"""Forgot Password functionality. - Starts"""
def password_forgot(request):
    reset_form = PasswordResetFormUnique()
    if request.method == 'GET':
        succes_msg = request.session.pop('succes_msg', None)
        return render(request, 'userProfile/forgot_password.html' , {'reset_form':reset_form,
                                                                     'succes_msg':succes_msg,})
    if request.method == 'POST':
        form = PasswordResetFormUnique(request.POST)
        if form.is_valid():
            response = auth_views.PasswordResetView.as_view(
                form_class=PasswordResetFormUnique,
                template_name='userProfile/login.html',
                email_template_name='userProfile/password_reset_email.html',
                success_url=reverse_lazy('userProfile:login'),
            )(request)
            # Check if the email was sent
            if response.status_code == 302:  # Redirect status code indicating success
                # The email was sent
                return HttpResponseRedirect(reverse_lazy('userProfile:login'))
            else:
                # The email sending failed
                return HttpResponse(f"Email sending failed. {response}")
        else:
            message = _("Invalid Email Provided")
            return render(request, 'userProfile/forgot_password.html', {'form': form,
                                                                        'message':message
                                                                        })
"""Forgot Password functionality. - Ends"""

"""My Profile functionality. - Starts"""
@login_required
def profile(request, profile_id=None):
    user_id = request.user.id
    roles = orphanageRoles.objects.all()
    if request.method == 'GET':
        if profile_id:
            try:
                profiledtls = Profile.objects.get(id=profile_id)
                user_create = 1 if profiledtls.user and profiledtls.user.id == user_id else 0
            except Exception as e:
                print(e)
        else:
            user_create = 1
            try:
                profiledtls = Profile.objects.get(user_id=user_id)
            except:
                profiledtls = None
        return render(request, 'userProfile/profile.html' , {
                                                            'profiledtls':profiledtls,
                                                            'roles':roles,
                                                            'user_create':user_create,
                                                              })
    if request.method == 'POST':
        if request.POST.get('role') == '#' and request.POST.get('new_role'):
            role = orphanageRoles.objects.create(role_name=request.POST.get('new_role'))
        else:
            try:
                role = orphanageRoles.objects.get(id=request.POST.get('role'))
            except Exception as e:
                print(e)
        role_id = role.id
        if request.POST.get('user_create'):
            User.objects.filter(id = user_id).update(first_name=request.POST.get('first_name'),
                                                    last_name=request.POST.get('last_name'),
                                                    email=request.POST.get('email'),
                                                   )
            if not Profile.objects.filter(user_id=user_id).exists():
                profile = Profile.objects.create(   user_id = user_id,
                                                    profile_name = request.POST.get('first_name')+' '+request.POST.get('last_name'),
                                                    role_id = role_id,
                                                    dob = request.POST.get('dob')
                                                )
            else:
                profile = Profile.objects.get(user_id=user_id)
                profile.profile_name = request.POST.get('first_name')+' '+request.POST.get('last_name')
                profile.role_id = role_id
                profile.dob = request.POST.get('dob')

        #For Update profile- end
        else:
            profile = Profile.objects.create(
                                            profile_name = request.POST.get('profile_name'),
                                            role_id = role_id,
                                            dob = request.POST.get('dob')
                )
        if profile:
            if request.FILES.get('profile_pic1'):
                profile.profile_photo1=request.FILES.get('profile_pic1')
            if request.FILES.get('profile_pic2'):
                profile.profile_photo2=request.FILES.get('profile_pic2')
            if request.FILES.get('profile_pic3'):
                profile.profile_photo3=request.FILES.get('profile_pic3')
            profile.save()
        if not request.POST.get('user_create'):
            return HttpResponseRedirect(reverse_lazy('userProfile:all_profiles'))
    return HttpResponseRedirect(reverse_lazy('mlpipeline:dashboard'))

"""Edit Profile functionality. - Ends"""

"""Create New Profile - Starts"""
@login_required
def new_profile(request):
    roles = orphanageRoles.objects.all()
    if request.method == 'GET':
        return render(request, 'userProfile/profile.html', {'roles':roles})
"""Create New Profile - Ends"""

"""Show all Profiles - Starts"""
def all_profiles(request):
    profiles = Profile.objects.all()
    return render(request, 'userProfile/all_profiles.html', {'profiles':profiles})
"""Show all Profiles - Ends"""

"""Delete Profile - Starts"""
@login_required
def delete_profile(request, profile_id):
    profile = get_object_or_404(Profile, pk=profile_id)
    if request.user.is_superuser:
        profile.delete()
        messages.success(request, 'Profile deleted successfully.')
    return HttpResponseRedirect(reverse_lazy('userProfile:all_profiles'))
"""Delete Profile - Ends"""
