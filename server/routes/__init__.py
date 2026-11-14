"""API 라우터 모음."""
from server.routes import chat, status, tools, watch

ROUTERS = [status.router, chat.router, tools.router, watch.router]
