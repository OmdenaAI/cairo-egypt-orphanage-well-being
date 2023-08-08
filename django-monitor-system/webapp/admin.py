from django.contrib import admin
from django_object_actions import DjangoObjectActions
from .models import *
from .CreateEncods import CreateEncods

class orphans_admin(DjangoObjectActions, admin.ModelAdmin):
    readonly_fields = ['image_tag']
    list_display = ['name', 'image_tag']
    ordering = ['name']

    def generate(modeladmin, request, queryset):
        encodings = CreateEncods()
        encodings.create_encodings(request)

    changelist_actions = ['generate']

admin.site.register(orphan_list, orphans_admin)
