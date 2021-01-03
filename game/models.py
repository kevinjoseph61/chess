from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Game(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE,related_name="owner")
    opponent = models.ForeignKey(User, on_delete=models.CASCADE,related_name="opponent", null=True)
    owner_side= models.CharField(max_length=10, default="white")
    owner_online = models.BooleanField(default=False)
    opponent_online = models.BooleanField(default=False)
    fen = models.CharField(max_length=92, null=True, blank=True)
    pgn = models.TextField(null=True, blank=True)
    winner = models.CharField(max_length=20, null=True, blank=True)
    level = models.CharField(max_length=15, null=True, blank=True)
    CHOICES=(
        (1,"Game Created. Waiting for opponent"),
        (2,"Game Started"),
        (3,"Game Ended"))
    status = models.IntegerField(default=1,choices=CHOICES)
