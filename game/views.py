from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Game
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views import View
from django.contrib.auth.models import User
from django.db.models import Q

@login_required
def index(request):
    l = Game.objects.all().filter(status=1).exclude(owner=request.user)
    pub = []
    for i in l:
        g = {}
        if i.owner_side == "white":
            g["side"] = "Black"
        else:
            g["side"] = "White"
        g["link"] = f"/game/{i.pk}"
        g["owner"] = i.owner
        g["level"] = i.level
        pub.append(g)
    return render(request, "game/lobby.html", {"public":pub})

@login_required
def game(request, game_id):
    game = get_object_or_404(Game,pk=game_id)
    if game.status == 3:
        messages.add_message(request, messages.ERROR, "This game has already been completed! Start another")
        return HttpResponseRedirect(reverse("lobby"))
    if request.user != game.owner:
        if game.opponent == None:
            game.opponent = request.user
            game.status = 2
            game.save()
            messages.add_message(request, messages.SUCCESS, "You have joined this game successfully")
        elif game.opponent != request.user:
            messages.add_message(request, messages.ERROR, "This game already has enough participants. Try joining another")
            return HttpResponseRedirect(reverse("lobby"))
    return render(request, "game/game.html", {"game_id":game_id})

@login_required
def single(request):
    return render(request, "game/single.html")

class createGame(LoginRequiredMixin, View):
    def get(self, request):
        return render(request,"game/create.html")
    def post(self, request):
        username = request.POST["username"]
        side = request.POST["side"]
        level = request.POST["level"]
        if username:
            try:
                u = User.objects.get(username=username)
                if u ==  request.user:
                    messages.add_message(request, messages.ERROR, "You can't play a game with yourself!")
                    return HttpResponseRedirect(reverse("create"))
                g = Game(owner=request.user, opponent=u, owner_side=side, status=2)
                g.save()
                return HttpResponseRedirect('/game/'+str(g.pk))
            except Exception as e:
                print(e)
                messages.add_message(request, messages.ERROR, "The username entered does not exist")
                return HttpResponseRedirect(reverse("create"))
        else:
            if level == "undef":
                messages.add_message(request, messages.ERROR, "Please choose a level if you are creating a public room!")
                return HttpResponseRedirect(reverse("create"))
            l = Game(owner=request.user, owner_side=side, level=level)
            l.save()
            messages.add_message(request, messages.SUCCESS, "Game created and displayed in Lobby. Check Ongoing Games to see status")
            return HttpResponseRedirect(reverse("lobby"))

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
        messages.add_message(request, messages.SUCCESS, "User successfully registered! Login now...")
        return HttpResponseRedirect("/accounts/login/")

@login_required
def ongoing(request):
    games = []
    l = Game.objects.all().filter(owner=request.user).filter(status=1)
    g = Game.objects.all().filter(Q(owner=request.user) | Q(opponent=request.user)).filter(status=2)
    for i in g:
        x = {}
        if i.owner == request.user:
            x["opponent"] = i.opponent
            x["side"] = i.owner_side
        else:
            x["opponent"] = i.owner
            if i.owner_side == "white":
                x["side"] = "black"
            else:
                x["side"] = "white"
        x["link"] = f"/game/{i.pk}"
        games.append(x) 
    return render(request, "game/ongoing.html", {"public":l, "ongoing": games})

@login_required
def completed(request):
    games=[]
    g = Game.objects.all().filter(Q(owner=request.user) | Q(opponent=request.user)).filter(status=3)
    for i in g:
        x = {}
        x["result"] = ""
        if i.owner == request.user:
            x["opponent"] = i.opponent
            x["side"] = i.owner_side
            if i.winner == "White wins":
                if i.owner_side == "white":
                    x["result"] = "You won this match"
                else:
                    x["result"] = "You lost this match"
            elif i.winner == "Black wins":
                if i.owner_side == "black":
                    x["result"] = "You won this match"
                else:
                    x["result"] = "You lost this match"
            else:
                x["result"] = i.winner
        else:
            x["opponent"] = i.owner
            if i.owner_side == "white":
                x["side"] = "black"
            else:
                x["side"] = "white"
            if i.winner == "Black wins":
                if i.owner_side == "white":
                    x["result"] = "You won this match"
                else:
                    x["result"] = "You lost this match"
            elif i.winner == "White wins":
                if i.owner_side == "black":
                    x["result"] = "You won this match"
                else:
                    x["result"] = "You lost this match"
            else:
                x["result"] = i.winner
        games.append(x)
    return render(request, "game/completed.html", {"completed": games})
