from django.db import models

# Create your models here.
class upload(models.Model):
    upload=models.FileField(upload_to="media")
    name = models.CharField(max_length=32)