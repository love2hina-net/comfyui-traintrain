import logging

from .scripts import gradio_server

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

WEB_DIRECTORY = "./js"
NODE_CLASS_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS']

logger.info("[TrainTrain] ComfyUI TrainTrain loaded")
