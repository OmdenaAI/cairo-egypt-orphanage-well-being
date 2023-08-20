from django.db import models
from django.contrib.auth.models import User
from django.db.models.deletion import CASCADE
from django.db.models.signals import pre_save
from django.dispatch import receiver

# Create your models here.

# Orphanage Roles model starts here
class orphanageRoles(models.Model):
    role_name = models.CharField(max_length=250,verbose_name="Orphanage Role Name")
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name="orphanageRolesCreatedBy")
    updated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name="orphanageRolesUpdatedBy")
    created_date = models.DateTimeField(auto_now_add=True,auto_now=False)
    updated_date = models.DateTimeField(auto_now_add=False, auto_now=True)

    def __str__(self):
        return self.role_name

    class Meta:
        verbose_name_plural = "Orphanage Roles"
        db_table = 'orphanage_roles'

# presave implemented with django signals - https://docs.djangoproject.com/en/4.2/topics/signals/
@receiver(pre_save, sender=orphanageRoles)
def set_created_updated_by(sender, instance, request=None, **kwargs):
    if not instance.pk:  # New instance
        instance.created_by = request.user if request else None
    else:
        instance.updated_by = request.user if request else None
# Orphanage Roles model ends here

# Profile model starts from here
class Profile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="User Id", blank=True, null=True)
    profile_name = models.CharField(max_length=250,verbose_name="Profile Name")
    role = models.ForeignKey(orphanageRoles, on_delete=models.CASCADE, verbose_name="Orphanage Roles Id")
    dob = models.DateTimeField()
    profile_photo1 = models.ImageField(upload_to='profile_photo/', blank=True, null=True, verbose_name="Profile Photo 1")
    profile_photo2 = models.ImageField(upload_to='profile_photo/', blank=True, null=True, verbose_name="Profile Photo 2")
    profile_photo3 = models.ImageField(upload_to='profile_photo/', blank=True, null=True, verbose_name="Profile Photo 3")
    encoded_photo1 = models.BinaryField(blank=True, null=True, editable=False, verbose_name="Profile Photo 1 Encoded")
    encoded_photo2 = models.BinaryField(blank=True, null=True, editable=False, verbose_name="Profile Photo 2 Encoded")
    encoded_photo3 = models.BinaryField(blank=True, null=True, editable=False, verbose_name="Profile Photo 3 Encoded")
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name="ProfileCreatedBy")
    updated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name="ProfileUpdatedBy")
    created_date = models.DateTimeField(auto_now_add=True,auto_now=False)
    updated_date = models.DateTimeField(auto_now_add=False, auto_now=True)

    def __str__(self):
        return self.profile_name

    class Meta:
        verbose_name_plural = "Profile"
        db_table = 'profile'

    def save(self, *args, **kwargs):
        try:
            obj = Profile.objects.get(pk=self.pk)
        except Profile.DoesNotExist:
            pass
        else:
            if not obj.profile_photo1 == self.profile_photo1:
                self.encoded_photo1 = b''
            if not obj.profile_photo2 == self.profile_photo2:
                self.encoded_photo2 = b''
            if not obj.profile_photo3 == self.profile_photo3:
                self.encoded_photo3 = b''
        super(Profile, self).save(*args, **kwargs)

@receiver(pre_save, sender=Profile)
def set_created_updated_by(sender, instance, **kwargs):
    if not instance.pk:  # New instance
        instance.created_by = instance.user
    else:
        instance.updated_by = instance.user
# Profile model ends from here
