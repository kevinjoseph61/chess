from django.urls import path, re_path

from channels.http import AsgiHandler
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

from game.consumers import GameConsumer, SingleConsumer

application = ProtocolTypeRouter({
    "websocket": AuthMiddlewareStack(
        URLRouter([
            path(r'game/<int:game_id>', GameConsumer.as_asgi()),
            path(r'single/', SingleConsumer.as_asgi())
        ]),
    ),
})
