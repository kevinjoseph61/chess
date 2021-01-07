"""
ASGI config for chess project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/asgi/
"""

import os
import django
from django.conf import settings
from channels.routing import get_default_application
from asgi_middleware_static_file import ASGIMiddlewareStaticFile

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pychess.settings')

django.setup()

application = get_default_application()

application = ASGIMiddlewareStaticFile(
    application, static_url=settings.STATIC_URL, static_paths=[settings.STATIC_ROOT]
)