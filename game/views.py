from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Game
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views import View
from django.contrib.auth.models import User

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
            messages.add_message(request, messages.SUCCESS, "You have joined this game successfully")
        elif game.opponent != request.user:
            messages.add_message(request, messages.ERROR, "This game already has enough participants. Try joining another")
            return HttpResponseRedirect(reverse("lobby"))
    return render(request, "game/game.html", {"game_id":game_id})

class createGame(LoginRequiredMixin, View):
    def get(self, request):
        return render(request,"game/create.html")
    def post(self, request):
        username = request.POST["username"]
        side = request.POST["side"]
        if username:
            try:
                u = User.objects.get(username=username)
                if u ==  request.user:
                    messages.add_message(request, messages.ERROR, "You can't play a game with yourself!")
                    return HttpResponseRedirect(reverse("create"))
                g = Game(owner=request.user, opponent=u, owner_side=side)
                g.save()
                return HttpResponseRedirect('/game/'+str(g.pk))
            except Exception as e:
                print(e)
                messages.add_message(request, messages.ERROR, "The username entered does not exist")
                return HttpResponseRedirect(reverse("create"))
        else:
            messages.add_message(request, messages.ERROR, "Enter a username")
            return HttpResponseRedirect(reverse("create"))

class register(View):
    def get(self, request):
        return render(request, "registration/signup.html")
    def post(self, request):
        first_name=request.POST["fname"]
        last_name=request.POST["lname"]
        username=request.POST["username"]
        email=request.POST["email"]
        password=request.POST["password"]
        passwordconf=request.POST["passwordconf"]
        if password != passwordconf:
            messages.add_message(request, messages.ERROR, "Passwords do not match")
            return HttpResponseRedirect(reverse("register"))
        try:
            User.objects.create_user(username=username, email=email, password=password, first_name=first_name, last_name=last_name)
        except:
            messages.add_message(request, messages.ERROR, "This username already exists!")
            return HttpResponseRedirect(reverse("register"))
        messages.add_message(request, messages.SUCCESS, "User successfully registered! Login...")
        return HttpResponseRedirect("/accounts/login/")