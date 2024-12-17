import asyncio
import threading

import gradio
from gradio.routes import App
from aiohttp import web
from server import PromptServer

class GradioServer:
    _instance = None

    INTERFACE: gradio.Interface = None
    APP: App = None
    LOCAL_URL: str = None
    SHARE_URL: str = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(GradioServer, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def launch(this) -> None:
        def ret(input: str) -> str:
            return f"Hello, {input}!"

        if not this.INTERFACE:
            this.INTERFACE = gradio.Interface(fn=ret, inputs="textbox", outputs="textbox")
            this.APP, this.LOCAL_URL, this.SHARE_URL = this.INTERFACE.launch(prevent_thread_lock=True)

    def close(this) -> None:
        if this.INTERFACE:
            this.INTERFACE.close()
            this.INTERFACE = None

@PromptServer.instance.routes.get("/traintrain/start_server")
def start_server(request: web.Request) -> web.Response:
    server = GradioServer()
    server.launch()
    return web.json_response(status=200, data={'url': server.LOCAL_URL})

@PromptServer.instance.routes.get("/traintrain/stop_server")
def stop_server(request: web.Request) -> web.Response:
    server = GradioServer()
    server.close()
    return web.json_response(status=200, data={})
