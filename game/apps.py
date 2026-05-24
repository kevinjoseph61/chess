from django.apps import AppConfig

class GameConfig(AppConfig):
    name = 'game'
    def ready(self):
        try:
            from .models import Game
            g = Game.objects.all()
            for i in g:
                i.opponent_online = i.owner_online = False
                i.save()
            print("Resetting online statuses")
        except Exception:
            pass

