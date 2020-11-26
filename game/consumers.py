from channels.generic.websocket import AsyncJsonWebsocketConsumer
from channels.db import database_sync_to_async
from .models import Game
from django.contrib.auth.models import User

room=0

class GameConsumer(AsyncJsonWebsocketConsumer):

    async def connect(self):
        if self.scope["user"].is_anonymous:
            await self.close()
            return
        self.game_id = self.scope['url_route']['kwargs']['game_id']
        try: 
            self.game_id=int(self.game_id)
        except:
            await self.close()
            return
        if await database_sync_to_async(self.verify(self.game_id)):
            pass
        else:
            await self.close()
            return
        await self.accept()

    async def receive_json(self, content):
        command = content.get("command", None)
        try:
            if command == "new-move":
                await self.new_move(content["source"],content["target"])
        except:
            pass

    async def disconnect(self, code):
        global room
        room-=1

    async def join_room(self, color):
        await self.channel_layer.group_add(
            "default",
            self.channel_name,
        )
        await self.send_json({
            "command":"join",
            "orientation": color,
        })

    async def new_move(self, source, target):
        await self.channel_layer.group_send(
            "default",
            {
                "type": "move.new",
                "source": source,
                "target": target,
                'sender_channel_name': self.channel_name
            }
        )
    
    async def move_new(self, event):
        if self.channel_name != event['sender_channel_name']:
            await self.send_json({
                "command":"new-move",
                "source": event["source"],
                "target": event["target"],
            })

    def verify(self, game_id):
        game = Game.objects.get(id=game_id)
        if not game:
            return False
        user = User.objects.get_by_natural_key(self.scope["user"].username)
        if game.opponent == user:
            game.opponent_online = True
        elif game.owner == user:
            game.owner_online == True
        else:
            return False
        game.save()
        return True
            

