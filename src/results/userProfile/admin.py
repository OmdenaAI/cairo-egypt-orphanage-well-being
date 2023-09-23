from django.contrib import admin
from import_export import resources
from import_export.fields import Field
from import_export.admin import ImportExportModelAdmin
from import_export.widgets import ForeignKeyWidget, DateTimeWidget, IntegerWidget
from .models import orphanageRoles, Profile
from django.contrib.auth.models import User

# Register your models here.

'''orphanageRoles Admin Starts Here'''
class orphanageRolesResource(resources.ModelResource):
    role_name = Field(column_name='role_name', attribute='role_name')

    class Meta:
        model = orphanageRoles
        fields = ('role_name',)
        export_order = fields
        import_id_fields = fields

class orphanageRolesAdmin(ImportExportModelAdmin, admin.ModelAdmin):
    resource_class = orphanageRolesResource
    list_display = ('role_name',)
    search_fields = ('role_name',)
    fields = ('role_name',)
'''orphanageRoles Ends Here'''

'''Profile Admin Starts Here'''
class ProfileResource(resources.ModelResource):
    user = Field(column_name='user',attribute='user',widget=ForeignKeyWidget(User, 'username'))
    profile_name = Field(column_name='profile_name', attribute='profile_name')
    role = Field(column_name='role',attribute='role',widget=ForeignKeyWidget(orphanageRoles, 'role_name'))
    dob = Field(column_name='dob',attribute='dob',widget=DateTimeWidget(format='%Y-%m-%d %H:%M:%S'))

    class Meta:
        model = Profile
        fields = ('user', 'profile_name', 'role', 'dob', 'profile_photo1', 'profile_photo2', 'profile_photo3', 'encoded_photo1', 'encoded_photo2', 'encoded_photo3',)
        export_order = fields
        import_id_fields = fields  # Adjust this according to your needs

class ProfileAdmin(ImportExportModelAdmin, admin.ModelAdmin):
    resource_class = ProfileResource
    list_display = ('user', 'profile_name', 'role', 'dob',)
    search_fields = ('user__username', 'profile_name','role__role_name')
    readonly_fields = ('encoded_photo1', 'encoded_photo2', 'encoded_photo3', 'created_by', 'created_date', 'updated_by', 'updated_date')
'''Profile Ends Here'''

admin.site.register(orphanageRoles, orphanageRolesAdmin)
admin.site.register(Profile, ProfileAdmin)
