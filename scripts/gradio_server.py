import gradio
import gradio.blocks
from gradio.routes import App
from aiohttp import web
from server import PromptServer

from . import traintrain

class GradioServer:
    _instance = None

    BLOCKS: gradio.Blocks = None
    APP: App = None
    LOCAL_URL: str = None
    SHARE_URL: str = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(GradioServer, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def launch(this) -> None:
        if not this.BLOCKS:
            this.BLOCKS, _, _ = traintrain.on_ui_tabs()
            this.APP, this.LOCAL_URL, this.SHARE_URL = this.BLOCKS.launch(prevent_thread_lock=True)

    def close(this) -> None:
        if this.BLOCKS:
            this.BLOCKS.close()
            this.BLOCKS = None

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
