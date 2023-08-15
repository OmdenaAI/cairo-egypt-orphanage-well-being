from django.db import models
from django.contrib.auth.backends import ModelBackend
from django.contrib.auth.models import User
import re
from django.contrib.auth import get_user_model


email_re = re.compile(
    r"(^[-!#$%&'*+/=?^_`{}|~0-9A-Z]+(\.[-!#$%&'*+/=?^_`{}|~0-9A-Z]+)*"
    r'|^"([\001-\010\013\014\016-\037!#-\[\]-\177]|\\[\001-\011\013\014\016-\177])*"'
    r')@(?:[A-Z0-9-]+\.)+[A-Z]{2,6}$', re.IGNORECASE)

phone_re = re.compile(r'^[123456789]\d{9}$')

class EmailAuthBackend(ModelBackend):
    '''Logs users using email addresses rather than username
    '''
    def authenticate(self, request,username=None, password=None):
        User = get_user_model()
        if email_re.search(username):
            try:
                user = User.objects.get(email=username)
                if user.check_password(password):
                    return user
            except User.DoesNotExist:
                return None
            
        if '@' not in username:
            try:
                user = User.objects.get(username=username)
                if user.check_password(password):
                    return user
            except User.DoesNotExist:
                return None
        
    def get_user(self, user_id):
       try:
          return User.objects.get(pk=user_id)
       except User.DoesNotExist:
          return None
