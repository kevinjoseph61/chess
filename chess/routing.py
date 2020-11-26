from django.urls import path, re_path

from channels.http import AsgiHandler
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

from game.consumers import GameConsumer

application = ProtocolTypeRouter({
    "websocket": AuthMiddlewareStack(
        URLRouter([
            re_path(r'game/(?P<game_id>\w+)/$', GameConsumer.as_asgi()),
        ]),
    ),
})
