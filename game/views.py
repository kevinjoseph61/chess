from django.shortcuts import render
from django.contrib.auth.decorators import login_required

@login_required
def index(request):
    return render(request, "game/lobby.html")

def game(request):
    return render(request, "game/game.html")
