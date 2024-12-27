from cProfile import label
from typing import Iterable, Any
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from gradio.blocks import Block
from PIL import Image, ImageChops
import random
import numpy as np
from . import comfyui
from .comfyui import ToolButton, create_refresh_button
#from modules import scripts, script_callbacks, sd_models, sd_vae
#from modules.shared import opts
#from modules.ui import create_output_panel, create_refresh_button
#from ..trainer import train, trainer, gen
from ..trainer import config as cfg, train, trainer
from ..trainer.config import ControlConfig
from packaging import version

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

jsonspath = trainer.JSONSPATH
logspath = trainer.LOGSPATH

MODEL_TYPES = [
    ("StableDiffusion v1", train.ModelType.SDv1),
    ("StableDiffusion v2", train.ModelType.SDv2),
    ("StableDiffusion XL", train.ModelType.SDXL)
]

R_COLUMN1 = (
    ControlConfig.NETWORK_TYPE,
    ControlConfig.NETWORK_RANK,
    ControlConfig.NETWORK_ALPHA,
    ControlConfig.LORA_DATA_DIRECTORY,
    ControlConfig.LORA_TRIGGER_WORD)
R_COLUMN2 = (
    ControlConfig.IMAGE_SIZE,
    ControlConfig.TRAIN_ITERATIONS,
    ControlConfig.TRAIN_BATCH_SIZE,
    ControlConfig.TRAIN_LEARNING_RATE)
R_COLUMN3 = (
    ControlConfig.TRAIN_OPTIMIZER,
    ControlConfig.TRAIN_LR_SCHEDULER,
    ControlConfig.SAVE_LORA_NAME,
    ControlConfig.USE_GRADIENT_CHECKPOINTING)
ROW1 = (ControlConfig.NETWORK_BLOCKS,)

O_COLUMN1 = (
    ControlConfig.NETWORK_CONV_RANK,
    ControlConfig.NETWORK_CONV_ALPHA,
    ControlConfig.NETWORK_ELEMENT,
    ControlConfig.IMAGE_BUCKETS_STEP,
    ControlConfig.IMAGE_MIN_LENGTH,
    ControlConfig.IMAGE_MAX_RATIO,
    ControlConfig.SUB_IMAGE_NUM,
    ControlConfig.IMAGE_MIRRORING,
    ControlConfig.IMAGE_USE_FILENAME_AS_TAG,
    ControlConfig.IMAGE_DISABLE_UPSCALE,
    ControlConfig.IMAGE_USE_TRANSPARENT_BACKGROUND_AJUST,
    ControlConfig.TRAIN_FIXED_TIMSTEPS_IN_BATCH)
O_COLUMN2 = (
    ControlConfig.TRAIN_TEXTENCODER_LEARNING_RATE,
    ControlConfig.TRAIN_SEED,
    ControlConfig.TRAIN_LR_STEP_RULES,
    ControlConfig.TRAIN_LR_WARMUP_STEPS,
    ControlConfig.TRAIN_LR_SCHEDULER_NUM_CYCLES,
    ControlConfig.TRAIN_LR_SCHEDULER_POWER, 
    ControlConfig.TRAIN_SNR_GAMMA,
    ControlConfig.SAVE_PER_STEPS)
O_COLUMN3 = (
    ControlConfig.TRAIN_MODEL_PRECISION,
    ControlConfig.TRAIN_LORA_PRECISION,
    ControlConfig.SAVE_PRECISION,
    ControlConfig.DIFF_LOAD_1ST_PASS,
    ControlConfig.DIFF_SAVE_1ST_PASS,
    ControlConfig.DIFF_1ST_PASS_ONLY,
    ControlConfig.LOGGING_SAVE_CSV,
    ControlConfig.LOGGING_VERBOSE,
    ControlConfig.SAVE_OVERWRITE,
    ControlConfig.SAVE_AS_JSON,
    ControlConfig.MODEL_V_PRED)

def mapping_ui(config: cfg.ConfigBase[cfg.ComponentConfig], sets: Iterable[ControlConfig]) -> dict:
    output = {}
    for i in sets:
        component = cfg.get_instance(config, i)
        output[component.elem_id] = component
        with gr.Row():
            component.render()

    return output

txt2img_gen_button = None
img2img_gen_button = None
paramsnames = []
txt2img_params = []
img2img_params = []

button_o_gen = None
button_t_gen = None
button_b_gen = None

prompts = None
imagegal_orig = None
imagegal_targ = None

def on_ui_tabs():
    global imagegal_orig, imagegal_targ, prompts, result
    global button_o_gen, button_t_gen, button_b_gen

    def list_presets() -> list[str]:
        json_files: list[str] = []
        with os.scandir(trainer.PRESETSPATH) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith('.json'):
                    json_files.append(entry.name.removesuffix(".json"))

        return json_files
    
    def load_preset(config: cfg.ConfigRoot[cfg.ComponentValue], components: dict) -> None:
        json_select = components[presets]
        filepath = os.path.join(trainer.PRESETSPATH, f"{json_select}.json")
        if os.path.exists(filepath):
            cfg.apply_dict(config, cfg.import_json(filepath))
        else:
            logger.error(f"[TrainTrain] Loading preset was failed, because file not found: {filepath}")

    def save_preset(config: cfg.ConfigRoot[cfg.ComponentValue], components: dict) -> dict[Block, Any]:
        cfg.export_json(trainer.PRESETSPATH, cfg.as_dict(config), False)
        return { presets: gr.update(choices=list_presets()) }

    folder_symbol = '\U0001f4c2'   
    load_symbol = '\u2199\ufe0f'   # ↙
    save_symbol = '\U0001f4be'     # 💾
    refresh_symbol = '\U0001f504'  # 🔄

    CONFIG_ROOT = cfg.ConfigRoot(cfg.ComponentConfig)
    cfg.create(CONFIG_ROOT)

    with gr.Blocks(css_paths=f"{comfyui.BASEDIR}/css/comfyui-traintrain.css") as ui:
        with gr.Tab("Train"):
            with gr.Row():
                with gr.Column():
                    start= gr.Button(value="Start Training",elem_classes=["compact_button"],variant='primary')
                with gr.Column():
                    stop= gr.Button(value="Stop",elem_classes=["compact_button"],variant='primary')
            with gr.Row():
                with gr.Column():
                    queue = gr.Button(value="Add to Queue", elem_classes=["compact_button"],variant='primary')
                with gr.Column():
                    stop_save= gr.Button(value="Stop and Save",elem_classes=["compact_button"],variant='primary')

            result = gr.Textbox(label="Message")

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        presets = gr.Dropdown(choices=list_presets(), show_label=False, elem_id="tt_preset", interactive=True)
                        loadpreset = ToolButton(value=load_symbol)
                        savepreset = ToolButton(value=save_symbol)
                        refleshpreset = ToolButton(value=refresh_symbol)
                with gr.Column():
                    with gr.Row():
                        sets_file = gr.Textbox(show_label=False)
                        openfolder = ToolButton(value=folder_symbol)
                        loadjson = ToolButton(value=load_symbol)
            with gr.Row():
                with gr.Column():
                    mode = CONFIG_ROOT.mode.instance
                    mode.render()
            with gr.Row():
                with gr.Column():
                    with gr.Row(equal_height=True):
                        model = CONFIG_ROOT.model.instance
                        model.render()
                        create_refresh_button(model, lambda: {}, lambda: {"choices": comfyui.list_checkpoints()}, "tt0:refresh_model")

                        # 初期表示項目の設定
                        checkpoints = comfyui.list_checkpoints()
                        default_checkpoint = None if len(checkpoints) <= 0 else checkpoints[0]
                        model.choices = checkpoints
                        model.value = default_checkpoint
                        ui.load(lambda: gr.update(choices=checkpoints, value=default_checkpoint), outputs=[model])
                with gr.Column():
                    with gr.Row(equal_height=True):
                        model_type = CONFIG_ROOT.model_type.instance
                        model_type.render()

                        # 初期表示項目の設定
                        model_type.choices = MODEL_TYPES
                        model_type.value = MODEL_TYPES[0][1]
                        ui.load(lambda: gr.update(choices=MODEL_TYPES, value=MODEL_TYPES[0][1]), outputs=[model_type])
                with gr.Column():
                    with gr.Row(equal_height=True):
                        vae = CONFIG_ROOT.vae.instance
                        vae.render()
                        create_refresh_button(vae, lambda: {}, lambda: {"choices": ["None"] + comfyui.list_vaes()}, "tt0:refresh_vae")

                        # 初期表示項目の設定
                        vaes = ["None"] + comfyui.list_vaes()
                        vae.choices = vaes
                        vae.value = vaes[0]
                        ui.load(lambda: gr.update(choices=vaes, value=vaes[0]), outputs=[vae])

            gr.HTML(value="Required Parameters(+prompt in iLECO, +images in Difference)")
            with gr.Row():
                with gr.Column(variant="compact"):
                    mapping_ui(CONFIG_ROOT, R_COLUMN1)
                with gr.Column(variant="compact"):
                    mapping_ui(CONFIG_ROOT, R_COLUMN2)
                with gr.Column(variant="compact"):
                    mapping_ui(CONFIG_ROOT, R_COLUMN3)
            with gr.Row():
                mapping_ui(CONFIG_ROOT, ROW1)

            gr.HTML(value="Option Parameters")
            with gr.Row():
                with gr.Column(variant="compact"):
                    mapping_ui(CONFIG_ROOT, O_COLUMN1)
                with gr.Column(variant="compact"):
                    mapping_ui(CONFIG_ROOT, O_COLUMN2)
                with gr.Column(variant="compact"):
                    mapping_ui(CONFIG_ROOT, O_COLUMN3)

            with gr.Accordion("2nd pass", open= False, visible = False) as diff_2nd:
                with gr.Row():
                    mapping_ui(CONFIG_ROOT, (ControlConfig.USE_2ND_PASS_SETTINGS,))
                    copy = gr.Button(value= "Copy settings from 1st pass")
                gr.HTML(value="Required Parameters")
                with gr.Row():  
                    with gr.Column(variant="compact"):
                        mapping_ui(CONFIG_ROOT.second_pass, R_COLUMN1)
                    with gr.Column(variant="compact"):
                        mapping_ui(CONFIG_ROOT.second_pass, R_COLUMN2)
                    with gr.Column(variant="compact"):
                        mapping_ui(CONFIG_ROOT.second_pass, R_COLUMN3)
                with gr.Row():
                    mapping_ui(CONFIG_ROOT.second_pass, ROW1)

                gr.HTML(value="Option Parameters")
                with gr.Row():
                    with gr.Column(variant="compact"):
                        mapping_ui(CONFIG_ROOT.second_pass, O_COLUMN1)
                    with gr.Column(variant="compact"):
                        mapping_ui(CONFIG_ROOT.second_pass, O_COLUMN2)
                    with gr.Column(variant="compact"):
                        mapping_ui(CONFIG_ROOT.second_pass, O_COLUMN3)

            with gr.Group(visible=False) as g_leco:
                prompts = mapping_ui(CONFIG_ROOT, (ControlConfig.ORIGINAL_PROMPT, ControlConfig.TARGET_PROMPT))
                with gr.Row():
                    neg_prompt = gr.TextArea(label="Negative Prompt(not userd in training)",lines=3)
                with gr.Row():
                    button_o_gen = gr.Button(value="Generate Original",elem_classes=["compact_button"],variant='primary')
                    button_t_gen = gr.Button(value="Generate Target",elem_classes=["compact_button"],variant='primary')
                    button_b_gen = gr.Button(value="Generate All",elem_classes=["compact_button"],variant='primary')
                with gr.Row():
                    with gr.Column():
                        pass
                        # o_g =  create_output_panel("txt2img", opts.outdir_txt2img_samples)
                        # imagegal_orig = [x for x in o_g] if isinstance(o_g, tuple) else [o_g.gallery, o_g.generation_info, o_g.infotext, o_g.html_log]
                    with gr.Column():
                        pass
                        # t_g =  create_output_panel("txt2img", opts.outdir_txt2img_samples)
                        # imagegal_targ = [x for x in t_g] if isinstance(t_g, tuple) else [t_g.gallery, t_g.generation_info, t_g.infotext, t_g.html_log]

            with gr.Group(visible=False) as g_diff:
                with gr.Row():
                    with gr.Column():
                        orig_image = gr.Image(label="Original Image", interactive=True)
                    with gr.Column():
                        targ_image = gr.Image(label="Target Image", interactive=True)

        with gr.Tab("Queue"):
            with gr.Row():
                reload_queue= gr.Button(value="Reload Queue",elem_classes=["compact_button"],variant='primary')
                delete_queue= gr.Button(value="Delete Queue",elem_classes=["compact_button"],variant='primary')
                delete_name= gr.Textbox(label="Name of Queue to delete")
            with gr.Row():
                queue_list = gr.DataFrame(headers=cfg.get_queue_header(CONFIG_ROOT))

        with gr.Tab("Plot"):
            with gr.Row():
                reload_plot= gr.Button(value="Reloat Plot",elem_classes=["compact_button"],variant='primary')
                plot_file = gr.Textbox(label="Name of logfile, blank for last or current training")
            with gr.Row():
                plot = gr.Plot()

        with gr.Tab("Image"):
            gr.HTML(value="Rotate random angle and scaling")
            image_result = gr.Textbox(label="Message")
            with gr.Row():
                with gr.Column(variant="compact"):
                    angle_bg= gr.Button(value="From Directory",elem_classes=["compact_button"],variant='primary')
                with gr.Column(variant="compact"):
                    angle_bg_i= gr.Button(value="From File",elem_classes=["compact_button"],variant='primary')
                with gr.Column(variant="compact"):
                    fix_side = gr.Radio(label="fix side", value= "none", choices =["none", "right", "left", "top", "bottom"] )
            with gr.Row():
                with gr.Column(variant="compact"):  
                    image_dir = gr.Textbox(label="Image directory")
                with gr.Column(variant="compact"):
                    output_name = gr.Textbox(label="Output name")
                with gr.Column(variant="compact"):
                    save_dir = gr.Textbox(label="Output directory")                   
            with gr.Row():
                num_of_images = gr.Slider(label="number of images", maximum=1000, minimum=0, step=1, value=5)
                max_tilting_angle = gr.Slider(label="max_tilting_angle", maximum=180, minimum=0, step=1, value=180)
                min_scale = gr.Slider(label="minimun downscale ratio", maximum=1, minimum=0, step=0.01, value=0.4)
            with gr.Row():
                change_angle = gr.Checkbox(label="change angle", value= False)
                change_scale = gr.Checkbox(label="change scale", value= False)

            input_image = gr.Image(label="Input Image", interactive=True, type="pil", image_mode="RGBA")

        dtrue = gr.Checkbox(value = True, visible= False)
        dfalse = gr.Checkbox(value = False, visible= False)

        # prompts |= { f"{neg_prompt.elem_id}": neg_prompt }
        # in_images = { orig_image, targ_image }

        angle_bg.click(change_angle_bg,[dtrue, image_dir, save_dir, input_image,output_name, num_of_images ,change_angle,max_tilting_angle, change_scale, min_scale, fix_side], [image_result])
        angle_bg_i.click(change_angle_bg,[dfalse, image_dir, save_dir, input_image,output_name, num_of_images ,change_angle,max_tilting_angle, change_scale, min_scale, fix_side], [image_result])

        cfg.event_proxy(CONFIG_ROOT, start.click,
                        lambda config, appends: train.train(config, *appends),
                        [neg_prompt, orig_image, targ_image], result)
        cfg.event_proxy(CONFIG_ROOT, queue.click,
                        lambda config, appends: train.queue(config, *appends),
                        [neg_prompt, orig_image, targ_image], result)
        cfg.event_proxy(CONFIG_ROOT, savepreset.click, save_preset, None, { presets })
        refleshpreset.click(lambda : gr.update(choices=list_presets()), outputs = presets)

        reload_queue.click(lambda: cfg.list_queues(CONFIG_ROOT, train.get_del_queue_list()), outputs=queue_list)
        delete_queue.click(lambda name: cfg.list_queues(CONFIG_ROOT ,train.get_del_queue_list(name)), inputs=[delete_name], outputs=queue_list)

        stop.click(train.stop_time,[dfalse],[result])
        stop_save.click(train.stop_time,[dtrue],[result])

        def change_the_mode(config: cfg.ConfigRoot[cfg.ComponentValue], components: dict) -> dict[Block, Any]:
            updates = {}
            mode = cfg.MODES.index(config.mode.value)

            # 通常コントロール分
            for component in config.components(cfg.ComponentValue):
                component.update = gr.update(visible=component.config.ENABLE[mode])
            # diff_2nd
            updates[diff_2nd] = gr.update(visible=((False, False, True)[mode]))
            # g_leco
            updates[g_leco] = gr.update(visible=((False, True, False)[mode]))
            # g_diff
            updates[g_diff] = gr.update(visible=((False, False, True)[mode]))
            return updates

        def change_the_block(type, select):
            blocks = cfg.BLOCKID17 if type == cfg.NETWORK_TYPES[0] else cfg.BLOCKID26
            return gr.update(choices = blocks, value = [x for x in select if x in blocks])

        def openfolder_f():
            os.startfile(jsonspath)

        # loadjson.click(trainer.import_json,[sets_file], [mode, model, vae] +  train_settings_1 +  train_settings_2 + prompts[:2])
        cfg.event_proxy(CONFIG_ROOT, loadpreset.click, load_preset, { presets }, None)
        cfg.event_proxy(CONFIG_ROOT, mode.change, change_the_mode, None, { diff_2nd, g_leco, g_diff })
        openfolder.click(openfolder_f)
        # copy.click(lambda *x: x, train_settings_1[1:], train_settings_2[1:])

        reload_plot.click(plot_csv, [plot_file],[plot])

    return (ui, "TrainTrain", "TrainTrain")

def plot_csv(csv_path):
    def get_csv(csv_path):
        csv_path = csv_path if ".csv" in csv_path else csv_path + ".csv"
        if csv_path:
            for root, dirs, files in os.walk(logspath):
                if csv_path in files:
                    return os.path.join(root, csv_path)

        # 指定されたファイル名が見つからない場合、または csv_path が空の場合
        # ディレクトリ内で最新の CSV ファイルを探す
        latest_csv = None
        latest_time = 0

        for root, dirs, files in os.walk(logspath):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    file_time = os.path.getmtime(file_path)

                    if file_time > latest_time:
                        latest_csv = file_path
                        latest_time = file_time

        return latest_csv

    df = pd.read_csv(get_csv(csv_path))
    x = df.columns[0]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 主要な y 軸 (2 列目)
    color = 'tab:red'
    ax1.set_xlabel(x)
    ax1.set_ylabel(df.columns[1], color=color)
    ax1.plot(df[x], df[df.columns[1]], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # 追加の y 軸 (3 列目以降)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Learning Rates', color=color)  # 他の列のラベル
    for column in df.columns[2:]:
        ax2.plot(df[x], df[column], label=column)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Training Result")
    fig.tight_layout()
    plt.legend()
    plt.grid(True)

    return plt.gcf()


# ここに必要な追加の関数を定義します。
def downscale_image(image, min_scale, fix_side=None):
    import random
    from PIL import Image

    scale = random.uniform(min_scale, 1)
    original_size = image.size
    new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    downscaled_image = image.resize(new_size, Image.ANTIALIAS)
    new_image = Image.new("RGBA", original_size, (0, 0, 0, 0))

    # 配置位置を決定する
    if fix_side == "right":
        x_position = original_size[0] - new_size[0]
        y_position = random.randint(0, original_size[1] - new_size[1])
    elif fix_side == "top":
        x_position = random.randint(0, original_size[0] - new_size[0])
        y_position = 0
    elif fix_side == "left":
        x_position = 0
        y_position = random.randint(0, original_size[1] - new_size[1])
    elif fix_side == "bottom":
        x_position = random.randint(0, original_size[0] - new_size[0])
        y_position = original_size[1] - new_size[1]
    else:
        # fix_sideがNoneまたは無効な値の場合、ランダムな位置に配置
        x_position = random.randint(0, original_size[0] - new_size[0])
        y_position = random.randint(0, original_size[1] - new_size[1])

    new_image.paste(downscaled_image, (x_position, y_position))
    return new_image


MARGIN = 5

def marginer(bbox, image):
    return (
        max(bbox[0] - MARGIN, 0),  # 左
        max(bbox[1] - MARGIN, 0),  # 上
        min(bbox[2] + MARGIN, image.width),  # 右
        min(bbox[3] + MARGIN, image.height)  # 下
    )


def change_angle_bg(from_dir, image_dir, save_dir, input_image, output_name, num_of_images ,
                                change_angle, max_tilting_angle, change_scale, min_scale, fix_side):

    if from_dir:
        image_files = [file for file in os.listdir(image_dir) if file.endswith((".png", ".jpg", ".jpeg"))]
    else:
        image_files = [input_image]

    for file in image_files:
        if isinstance(file, str):
            modified_folder_path = os.path.join(image_dir, "modified")
            os.makedirs(modified_folder_path, exist_ok=True)

            path = os.path.join(image_dir, file)
            name, extention = os.path.splitext(file)
            with Image.open(path) as img:
                img = img.convert("RGBA")
        else:
            modified_folder_path = save_dir
            os.makedirs(modified_folder_path, exist_ok=True)

            img = file
            name = output_name
            extention = "png"

        for i in range(num_of_images):
            modified_img = img

            #画像を回転
            if change_angle:
                angle = random.uniform(-max_tilting_angle, max_tilting_angle)
                modified_img = modified_img.rotate(angle, expand=True)

            if change_scale:
                modified_img = downscale_image(modified_img, min_scale, fix_side)

            # 変更した画像を保存
            save_path = os.path.join(modified_folder_path, f"{name}_id_{i}.{extention}")
            modified_img.save(save_path)

    return f"Images saved in {modified_folder_path}"


BLOCKID=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11","Not Merge"]
BLOCKIDXL=['BASE', 'IN0', 'IN1', 'IN2', 'IN3', 'IN4', 'IN5', 'IN6', 'IN7', 'IN8', 'M', 'OUT0', 'OUT1', 'OUT2', 'OUT3', 'OUT4', 'OUT5', 'OUT6', 'OUT7', 'OUT8', 'VAE']
BLOCKIDXLL=['BASE', 'IN00', 'IN01', 'IN02', 'IN03', 'IN04', 'IN05', 'IN06', 'IN07', 'IN08', 'M00', 'OUT00', 'OUT01', 'OUT02', 'OUT03', 'OUT04', 'OUT05', 'OUT06', 'OUT07', 'OUT08', 'VAE']
ISXLBLOCK=[True, True, True, True, True, True, True, True, True, True, False, False, False, True, True, True, True, True, True, True, True, True, True, False, False, False]

def getjsonlist():
    if not os.path.isdir(jsonspath):
        return []
    json_files = [f for f in os.listdir(jsonspath) if f.endswith('.json')]
    json_files = [f.replace(".json", "") for f in json_files]
    return json_files

# class GenParamGetter(scripts.Script):
#     events_assigned = False
#     def title(self):
#         return "TrainTrain Generation Parameter Getter"
    
#     def show(self, is_img2img):
#         return scripts.AlwaysVisible

#     def get_wanted_params(params,wanted):
#         output = []
#         for target in wanted:
#             if target is None:
#                 output.append(params[0])
#                 continue
#             for param in params:
#                 if hasattr(param,"label"):
#                     if param.label == target:
#                         output.append(param)
#         return output

#     def after_component(self, component: gr.components.Component, **_kwargs):
#         """Find generate button"""
#         if component.elem_id == "txt2img_generate":
#             GenParamGetter.txt2img_gen_button = component
#         elif  component.elem_id == "img2img_generate":
#             GenParamGetter.img2img_gen_button = component

#     def get_components_by_ids(root: gr.Blocks, ids: list[int]):
#         components: list[gr.Blocks] = []

#         if root._id in ids:
#             components.append(root)
#             ids = [_id for _id in ids if _id != root._id]
        
#         if hasattr(root,"children"):
#             for block in root.children:
#                 components.extend(GenParamGetter.get_components_by_ids(block, ids))
#         return components
    
#     def compare_components_with_ids(components: list[gr.Blocks], ids: list[int]):
#         return len(components) == len(ids) and all(component._id == _id for component, _id in zip(components, ids))

#     def get_params_components(demo: gr.Blocks, app):
#         global paramsnames, txt2img_params, img2img_params
#         for _id, _is_txt2img in zip([GenParamGetter.txt2img_gen_button._id, GenParamGetter.img2img_gen_button._id], [True, False]):
#             if hasattr(demo,"dependencies"):
#                 dependencies: list[dict] = [x for x in demo.dependencies if x["trigger"] == "click" and _id in x["targets"]]
#                 g4 = False
#             else:
#                 dependencies: list[dict] = [x for x in demo.config["dependencies"] if x["targets"][0][1] == "click" and _id in x["targets"][0]]
#                 g4 = True
            
#             dependency: dict = None

#             for d in dependencies:
#                 if len(d["outputs"]) == 4:
#                     dependency = d
            
#             if g4:
#                 params = [demo.blocks[x] for x in dependency['inputs']]
#                 if _is_txt2img:
#                     paramsnames = [x.label if hasattr(x,"label") else "None" for x in params]

#                 if _is_txt2img:
#                     txt2img_params = params
#                 else:
#                     img2img_params = params
#             else:
#                 params = [params for params in demo.fns if GenParamGetter.compare_components_with_ids(params.inputs, dependency["inputs"])]

#                 if _is_txt2img:
#                     paramsnames = [x.label if hasattr(x,"label") else "None" for x in params[0].inputs]

#                 if _is_txt2img:
#                     txt2img_params = params[0].inputs 
#                 else:
#                     img2img_params = params[0].inputs

#             # from pprint import pprint
#             # pprint(paramsnames)

#         if not GenParamGetter.events_assigned:
#             with demo:
#                 button_o_gen.click(
#                     fn=gen.setup_gen_p,
#                     inputs=[gr.Checkbox(value=False, visible=False), prompts[0], prompts[2], *txt2img_params],
#                     outputs=imagegal_orig,
#                 )

#                 button_t_gen.click(
#                     fn=gen.setup_gen_p,
#                     inputs=[gr.Checkbox(value=False, visible=False), prompts[1], prompts[2], *txt2img_params],
#                     outputs=imagegal_targ,
#                 )

#                 button_b_gen.click(
#                     fn=gen.gen_both,
#                     inputs=[*prompts, *txt2img_params],
#                     outputs=imagegal_orig + imagegal_targ
#                 )

#             GenParamGetter.events_assigned = True

#if __package__ == "traintrain":
#    script_callbacks.on_ui_tabs(on_ui_tabs)
#    script_callbacks.on_app_started(GenParamGetter.get_params_components)
