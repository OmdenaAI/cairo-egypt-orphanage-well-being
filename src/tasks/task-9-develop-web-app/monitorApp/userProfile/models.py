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

# presave implemented with django signals - https://docs.djangoproject.com/en/4.2/topics/signals/
@receiver(pre_save, sender=orphanageRoles)
def set_created_updated_by(sender, instance, **kwargs):
    if not instance.pk:  # New instance
        instance.created_by = instance.user
    else:
        instance.updated_by = instance.user
# Orphanage Roles model ends here

# class Rename:
#     def __init__(self, number):
#         self.number = number

#     def save(self, instance, filename):
#         instance = str(instance)
#         alphabet = instance[0].upper()
#         newpath = os.path.join(MEDIA_ROOT, alphabet)

#         if not os.path.exists(newpath):
#             os.makedirs(newpath)

#         ext = os.path.splitext(filename)[1]
#         filename = f"{instance}{self.number}{ext}"
#         return os.path.join(newpath, filename)

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

    # Code from django-monitor-app starts here
    # def fetch_img(self):
    #     local_host = 'http://127.0.0.1:8000'
    #     image_url = local_host + self.profile_photo1.url
    #     resp = urllib.request.urlopen(image_url)
    #     arr = np.asarray(bytearray(resp.read()), dtype="uint8")
    #     img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    #     return img

    # def set_filepath(self):
    #     path = os.path.join(MEDIA_ROOT, 'PhotoUI')
    #     filename = self.name + '_PhotoUI.jpg'
    #     filepath = os.path.join(path, filename)
    #     return filepath

    # def insert_text(self, img, filepath, text):
    #     image = imutils.resize(img, width=200)
    #     cv2.putText(image,text,(0, int(image.shape[0]/2)),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    #     cv2.imwrite(filepath, image)

    # def image_tag(self):
    #     if self.photo1:
    #         img = self.fetch_img()
    #         filepath = self.set_filepath()
    #         resized_img = imutils.resize(img, width=500)
    #         faceLocList = face_recognition.face_locations(resized_img)

    #         if len(faceLocList) == 1:
    #             faceLoc = faceLocList[0]
    #             y1,x2,y2,x1 = faceLoc
    #             cropped_img = resized_img[y1:y2, x1:x2]

    #             face_img = imutils.resize(cropped_img, width=200)
    #             cv2.imwrite(filepath, face_img)

    #         elif len(faceLocList) > 1:
    #             text = 'MANY FACES'
    #             self.insert_text(resized_img, filepath, text)

    #         else:
    #             text = 'NO FACE'
    #             self.insert_text(resized_img, filepath, text)

    #         self.photoUI = filepath
    #         self.save()

    #     return mark_safe('<img src="{}"/>'.format(self.photoUI.url))

    # image_tag.short_description = 'Face'

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
    # Code from django-monitor-app ends here
@receiver(pre_save, sender=Profile)
def set_created_updated_by(sender, instance, **kwargs):
    if not instance.pk:  # New instance
        instance.created_by = instance.user
    else:
        instance.updated_by = instance.user
# Profile model ends from here
