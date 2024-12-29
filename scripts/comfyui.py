import typing
import os.path as path
from functools import wraps

import gradio as gr
from gradio.components.base import Component

import folder_paths

refresh_symbol = '\U0001f504'  # 🔄

BASEDIR = path.normpath(path.join(path.dirname(path.abspath(__file__)), ".."))

def basedir() -> str:
    return BASEDIR

def list_checkpoints() -> list[str]:
    return folder_paths.get_filename_list("checkpoints")

def list_vaes() -> list[str]:
    return folder_paths.get_filename_list("vae")

def get_lora_dir() -> str:
    return folder_paths.get_folder_paths("loras")[0]

class FormComponent:
    webui_do_not_create_gradio_pyi_thank_you = True

    def get_expected_parent(self):
        return gr.components.Form

class ToolButton(gr.Button, FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    @wraps(gr.Button.__init__)
    def __init__(self, value="", *args, elem_classes=None, **kwargs):
        elem_classes = elem_classes or []
        super().__init__(*args, elem_classes=["tool", *elem_classes], value=value, **kwargs)

    def get_block_name(self):
        return "button"

type RefreshResult = typing.Sequence[dict[str, typing.Any]] | dict[str, typing.Any]

def create_refresh_button(components: typing.Sequence[Component] | Component,
                          callcback: typing.Callable[[], RefreshResult]) -> ToolButton:
    refresh_components = components if isinstance(components, typing.Sequence) else [components]

    def refresh() -> list[dict[str, typing.Any]]:
        updates: list[dict[str, typing.Any]] = [{} for _ in refresh_components]

        refresh_args = callcback()
        refresh_args = refresh_args if isinstance(refresh_args, typing.Sequence) else [refresh_args]

        if len(refresh_args) == 0:
            pass
        elif (len(refresh_components) == 1) and (len(refresh_args) > 1):
            raise ValueError("Expected single outputs, but got multiple outputs.")
        else:
            demultiplexer: bool = (len(refresh_components) == 1)

            for i, comp in enumerate(refresh_components):
                index = 0 if demultiplexer else i
                for key, value in refresh_args[index].items():
                    setattr(comp, key, value)
                updates[i] = gr.update(**refresh_args[index])
        
        return updates[0] if len(updates) == 1 else updates

    refresh_button = ToolButton(value=refresh_symbol)
    refresh_button.click(fn=refresh, inputs=[], outputs=refresh_components)
    return refresh_button
