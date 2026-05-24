from django.urls import path
from django.core.asgi import get_asgi_application

from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

from game.consumers import GameConsumer, SingleConsumer

application = ProtocolTypeRouter(
    {
        "http": get_asgi_application(),
        "websocket": AuthMiddlewareStack(
            URLRouter(
                [
                    path(r"game/<int:game_id>", GameConsumer.as_asgi()),
                    path(r"single/", SingleConsumer.as_asgi()),
                ]
            ),
        ),
    }
)
