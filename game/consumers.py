from channels.generic.websocket import AsyncJsonWebsocketConsumer, JsonWebsocketConsumer
from channels.db import database_sync_to_async
from .models import Game
from django.contrib.auth.models import User
from .chessAI import call_AI


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
        side = await self.verify(self.game_id)
        if side == False:
            await self.close()
            return
        await self.accept()
        await self.join_room(side)
        if side[2]:
            await self.opp_online()

    async def receive_json(self, content):
        command = content.get("command", None)
        try:
            if command == "new-move":
                await self.new_move(content["source"],content["target"],content["fen"],content["pgn"])
            elif command == "game-over":
                await self.game_over(content["result"])
            elif command == "resign":
                await self.resign()
                await self.game_over(content["result"])
        except:
            pass

    async def disconnect(self, code):
        await self.disconn()
        await self.opp_offline()

    async def join_room(self, data):
        await self.channel_layer.group_add(
            str(self.game_id),
            self.channel_name,
        )
        await self.send_json({
            "command":"join",
            "orientation": data[0],
            "pgn": data[1],
            "opp_online": data[2]
        })

    async def opp_offline(self):
        await self.channel_layer.group_send(
            str(self.game_id),
            {
                "type": "offline.opp",
                'sender_channel_name': self.channel_name
            }
        )
    
    async def offline_opp(self,event):
        if self.channel_name != event['sender_channel_name']:
            await self.send_json({
                "command":"opponent-offline",
            })
            print("sending offline")

    async def opp_online(self):
        await self.channel_layer.group_send(
            str(self.game_id),
            {
                "type": "online.opp",
                'sender_channel_name': self.channel_name
            }
        )
    
    async def online_opp(self,event):
        if self.channel_name != event['sender_channel_name']:
            await self.send_json({
                "command":"opponent-online",
            })

    async def resign(self):
        await self.channel_layer.group_send(
            str(self.game_id),
            {
                "type": "resign.game",
                'sender_channel_name': self.channel_name
            }
        )
    
    async def resign_game(self,event):
        if self.channel_name != event['sender_channel_name']:
            await self.send_json({
                "command":"opponent-resigned",
            })

    async def new_move(self, source, target, fen, pgn):
        await self.channel_layer.group_send(
            str(self.game_id),
            {
                "type": "move.new",
                "source": source,
                "target": target,
                "fen": fen,
                "pgn": pgn,
                'sender_channel_name': self.channel_name
            }
        )
    
    async def move_new(self, event):
        if self.channel_name != event['sender_channel_name']:
            await self.send_json({
                "command":"new-move",
                "source": event["source"],
                "target": event["target"],
                "fen": event["fen"],
                "pgn": event["pgn"],
            })
        await self.update(event["fen"],event["pgn"])

    @database_sync_to_async
    def game_over(self, result):
        game = Game.objects.all().filter(id=self.game_id)[0]
        if game.status == 3:
            return
        game.winner = result
        game.status = 3
        game.save()

    @database_sync_to_async
    def verify(self, game_id):
        game = Game.objects.all().filter(id=game_id)[0]
        if not game:
            return False
        user = self.scope["user"]
        side = "white"
        opp=False
        if game.opponent == user:
            game.opponent_online = True
            if game.owner_side == "white":
                side = "black"
            else:
                side = "white"
            if game.owner_online == True:
                opp = True
            print("Setting opponent online")
        elif game.owner == user:
            game.owner_online = True
            if game.owner_side == "white":
                side = "white"
            else:
                side = "black"
            if game.opponent_online == True:
                opp = True
            print("Setting owner online")
        else:
            return False
        game.save()
        return [side,game.pgn,opp]

    @database_sync_to_async
    def disconn(self):
        user = self.scope["user"]
        game = Game.objects.all().filter(id=self.game_id)[0]
        if game.opponent == user:
            game.opponent_online = False
            print("Setting opponent offline")
        elif game.owner == user:
            game.owner_online = False
            print("Setting owner offline")
        game.save()
        
    @database_sync_to_async
    def update(self, fen, pgn):
        game = Game.objects.all().filter(id=self.game_id)[0]
        if not game:
            print("Game not found")
            return
        game.fen = fen
        game.pgn = pgn
        game.save()
        print("Saving game details")


class SingleConsumer(JsonWebsocketConsumer):
    def connect(self):
        if self.scope["user"].is_anonymous:
            self.close()
            return
        self.accept()
        self.send_json({"command":"join", "orientation": "white"})

    def receive_json(self, content):
        command = content.get("command", None)
        try:
            if command == "new-move":
                move = call_AI(content["pgn"], int(content["level"]))
                print(move)
                self.send_json({"command": "new-move", "move": move})
            else: 
                pass
        except Exception as e:
            print(f"Error: {e}")

    def disconnect(self, code):
        pass