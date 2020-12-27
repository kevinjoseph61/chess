from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Game
from django.http import HttpResponseRedirect
from django.urls import reverse

@login_required
def index(request):
    return render(request, "game/lobby.html")

@login_required
def game(request, game_id):
    game = get_object_or_404(Game,pk=game_id)
    if request.user != game.owner:
        if game.opponent == None:
            game.opponent = request.user
            game.save()
        elif game.opponent != request.user:
            messages.add_message(request, messages.ERROR, "This game already has enough participants. Try joining another")
            return HttpResponseRedirect(reverse("index"))
    return render(request, "game/game.html", {"game_id":game_id})

