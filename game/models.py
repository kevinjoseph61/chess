from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Game(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE,related_name="owner")
    opponent = models.ForeignKey(User, on_delete=models.CASCADE,related_name="opponent")
    owner_side= models.CharField(max_length=10, default="white")
    owner_online = models.BooleanField(default=False)
    opponent_online = models.BooleanField(default=False)
    fen = models.CharField(max_length=92)
    pgn = models.TextField()
    winner = models.ForeignKey(User, on_delete=models.CASCADE,related_name="winner", null=True)
    CHOICES=(
        (1,"Game Created. Waiting for opponent"),
        (2,"Game Started"),
        (3,"Game Ended"))
    status = models.IntegerField(default=1,choices=CHOICES)