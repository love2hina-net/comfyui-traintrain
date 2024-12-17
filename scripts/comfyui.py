from functools import wraps
import gradio as gr

import folder_paths

refresh_symbol = '\U0001f504'  # 🔄

def list_checkpoints() -> list[str]:
    return folder_paths.get_filename_list("checkpoints")

def list_vaes() -> list[str]:
    return folder_paths.get_filename_list("vae")

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

def create_refresh_button(refresh_component, refresh_method, refreshed_args, elem_id) -> ToolButton:
    refresh_components = refresh_component if isinstance(refresh_component, list) else [refresh_component]

    label = None
    for comp in refresh_components:
        label = getattr(comp, 'label', None)
        if label is not None:
            break

    def refresh():
        refresh_method()
        args = refreshed_args() if callable(refreshed_args) else refreshed_args

        for k, v in args.items():
            for comp in refresh_components:
                setattr(comp, k, v)

        return [gr.update(**(args or {})) for _ in refresh_components] if len(refresh_components) > 1 else gr.update(**(args or {}))

    refresh_button = ToolButton(value=refresh_symbol, elem_id=elem_id)
    refresh_button.click(
        fn=refresh,
        inputs=[],
        outputs=refresh_components
    )
    return refresh_button
