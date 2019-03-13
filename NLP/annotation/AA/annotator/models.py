from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

class Speaker(models.Model):
    name = models.CharField(max_length=500)

    def __str__(self):
        return self.name

class Intent(models.Model):
    author = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    modified_date = models.DateField(default=timezone.now)
    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name

class Domain(models.Model):
    author = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    modified_date = models.DateField(default=timezone.now)
    name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.name

class Dialog(models.Model):
    name = models.CharField(max_length=2000)
    author = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    modified_date = models.DateField(default=timezone.now)
    
    class Meta(object):
        ordering = ["-modified_date"]

    def __str__(self):
        return self.name

class Sentence(models.Model):
    author = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    modified_date = models.DateField(default=timezone.now)
    text = models.TextField(max_length=5000)
    speaker = models.ForeignKey(Speaker, on_delete=models.CASCADE)
    domain = models.ForeignKey(Domain, on_delete=models.SET_NULL, null=True, blank=True)
    intent = models.ForeignKey(Intent, on_delete=models.SET_NULL, null=True, blank=True)
    dialog = models.ForeignKey(Dialog, on_delete=models.CASCADE, related_name='sentences')

    def __str__(self):
        return self.text

class DialogActValue(models.Model):
    author = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    act_type = models.CharField(max_length=255, unique=True)

    def __str__(self):
        return self.act_type

class DialogAct(models.Model):
    act_type = models.ForeignKey(DialogActValue, on_delete=models.CASCADE)
    sentence = models.ForeignKey(Sentence, on_delete=models.CASCADE, related_name='dialogActs')

    def __str__(self):
        return self.act_type.act_type

class Slot(models.Model):
    slot_type = models.CharField(max_length=1000)
    slot_value = models.CharField(max_length=1000, blank=True)
    dialogAct = models.ForeignKey(DialogAct, on_delete=models.CASCADE, related_name='slots')

    def __str__(self):
        return self.slot_type + '=\'' + self.slot_value + '\'' 

