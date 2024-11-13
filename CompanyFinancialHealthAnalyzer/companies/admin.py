from django.contrib import admin

# Register your models here.
from .models import Users, Companies 
admin.site.register(Users)