from django.urls import path
from userProfile import views as userProfile_views
from django.contrib.auth.views import PasswordResetConfirmView,PasswordResetCompleteView
from userProfile.forms import CustomSetPasswordForm

app_name = 'userProfile'

urlpatterns = [
    path('signup/', userProfile_views.signup_view, name="signup"),
    path('login/', userProfile_views.login_view, name="login"),
    path('logout/', userProfile_views.logout_view, name="logout"),
    path('change_password/', userProfile_views.change_password, name="change_password"),
    path('password_forgot/', userProfile_views.password_forgot, name='password_forgot'),
    path('password_reset_<uidb64>_<token>/',PasswordResetConfirmView.as_view(
                                             form_class=CustomSetPasswordForm,
                                             template_name='userProfile/password_reset_confirm.html',
                                             success_url='/userProfile/reset/done/'),
                                             name='password_reset_confirm'),
    path('reset/done/', PasswordResetCompleteView.as_view(
                                                template_name='userProfile/password_reset_complete.html'
                                                ), name='password_reset_complete'),
    path('profile/', userProfile_views.profile, name='profile'),
    path('profile/<int:profile_id>/', userProfile_views.profile, name='profile_with_id'),
    path('new_profile/', userProfile_views.new_profile, name='new_profile'),
    path('all_profiles/', userProfile_views.all_profiles, name='all_profiles'),
    path('delete_profile/<int:profile_id>/', userProfile_views.delete_profile, name='delete_profile')
    ]

