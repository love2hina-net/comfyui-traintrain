import os
import copy
import json
import typing
import logging
import numbers
from enum import Enum, StrEnum
from datetime import datetime
from dataclasses import dataclass

import gradio as gr
from gradio.components.base import Component
from gradio.blocks import Block

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MODES = ["LoRA", "iLECO", "Difference"]

BLOCKID26=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKID17=["BASE","IN01","IN02","IN04","IN05","IN07","IN08","M00","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKID12=["BASE","IN04","IN05","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05"]
BLOCKID20=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08"]

PRECISION_TYPES = ["fp32", "bf16", "fp16", "float32", "bfloat16", "float16"]
NETWORK_TYPES = ["lierla", "c3lier","loha"]
NETWORK_DIMS: list[int] = [2**x for x in range(10)]
NETWORK_ELEMENTS = ["Full", "CrossAttention", "SelfAttention"]
IMAGESTEPS = [x*64 for x in range(10)]
OPTIMIZERS = ["adamw", "adamw8bit","adafactor","lion", "prodigy", "dadaptadam","dadaptlion","adam8bit","adam",]
LOSS_REDUCTIONS = ["none", "mean"]

SCHEDULERS = ["linear", "cosine", "cosine_with_restarts" ,"polynomial", "constant", "constant_with_warmup" ,"piecewise_constant"]
#NOISE_SCHEDULERS = ["Euler A", "DDIM", "DDPM", "LMSD"]
NOISE_SCHEDULERS = ["DDIM", "DDPM", "PNDM", "LMS", "Euler", "Euler a", "DPMSolver", "DPMsingle", "Heun", "DPM 2", "DPM 2 a"]
TARGET_MODULES = ["Both", "U-Net", "Text Encoder"]

class EnableConfig(tuple[bool, bool, bool, bool], Enum):
    ALL     = True, True, True, True
    LORA    = True, False, False, False
    ILECO   = False, True, False, False
    NDIFF   = True, True, False, False
    DIFF    = False, False, True, True
    DIFF1   = False, False, True, False
    DIFF2   = False, False, False, True
    NDIFF2  = True, True, True, False

class ControlType(StrEnum):
    DROPDOWN = "DD"
    TEXTBOX = "TX"
    TEXTAREA = "TA"
    CHECKBOX = "CH"
    CHECKBOXGROUP = "CB"
    RADIO = "RD"

class ControlConfig(Enum):
    #root parameters
    MODE                    = ("mode", ControlType.RADIO, MODES, MODES[0], str, EnableConfig.ALL)
    MODEL                   = ("model", ControlType.DROPDOWN, None, None, str, EnableConfig.ALL)
    MODEL_TYPE              = ("model_type", ControlType.DROPDOWN, None, None, str, EnableConfig.ALL)
    VAE                     = ("vae", ControlType.DROPDOWN, None, None, str, EnableConfig.ALL)
    ORIGINAL_PROMPT         = ("original prompt", ControlType.TEXTAREA, None, "", str, EnableConfig.ALL)
    TARGET_PROMPT           = ("target prompt", ControlType.TEXTAREA, None, "", str, EnableConfig.ALL)
    USE_2ND_PASS_SETTINGS   = ("use_2nd_pass_settings", ControlType.CHECKBOX, None, False, bool, EnableConfig.DIFF2)

    #requiered parameters
    LORA_DATA_DIRECTORY     = ("lora_data_directory", ControlType.TEXTBOX, None, "", str, EnableConfig.LORA)
    LORA_TRIGGER_WORD       = ("lora_trigger_word", ControlType.TEXTBOX, None, "", str, EnableConfig.LORA)
    NETWORK_TYPE            = ("network_type", ControlType.DROPDOWN, NETWORK_TYPES, NETWORK_TYPES[0], str, EnableConfig.ALL)
    NETWORK_RANK            = ("network_rank", ControlType.DROPDOWN, NETWORK_DIMS[2:], 16, int, EnableConfig.ALL)
    NETWORK_ALPHA           = ("network_alpha", ControlType.DROPDOWN, NETWORK_DIMS, 8, int, EnableConfig.ALL)
    NETWORK_ELEMENT         = ("network_element", ControlType.DROPDOWN, NETWORK_ELEMENTS, None, str, EnableConfig.ALL)
    IMAGE_SIZE              = ("image_size(height, width)", ControlType.TEXTBOX, None, 512, str, EnableConfig.NDIFF)
    TRAIN_ITERATIONS        = ("train_iterations", ControlType.TEXTBOX, None, 1000, int, EnableConfig.ALL)
    TRAIN_BATCH_SIZE        = ("train_batch_size", ControlType.TEXTBOX, None, 2, int, EnableConfig.ALL)
    TRAIN_LEARNING_RATE     = ("train_learning_rate", ControlType.TEXTBOX, None, "1e-4", float, EnableConfig.ALL)
    TRAIN_OPTIMIZER         = ("train_optimizer", ControlType.DROPDOWN, OPTIMIZERS, "adamw", str, EnableConfig.ALL)
    TRAIN_LR_SCHEDULER      = ("train_lr_scheduler", ControlType.DROPDOWN, SCHEDULERS, "cosine", str, EnableConfig.ALL)
    SAVE_LORA_NAME          = ("save_lora_name", ControlType.TEXTBOX, None, "", str, EnableConfig.NDIFF2)
    USE_GRADIENT_CHECKPOINTING = ("use_gradient_checkpointing", ControlType.CHECKBOX, None, False, bool, EnableConfig.ALL)

    #option parameters
    NETWORK_CONV_RANK       = ("network_conv_rank", ControlType.DROPDOWN, [0] + NETWORK_DIMS[2:], 0, int, EnableConfig.ALL)
    NETWORK_CONV_ALPHA      = ("network_conv_alpha", ControlType.DROPDOWN, [0] + NETWORK_DIMS, 0, int, EnableConfig.ALL)
    TRAIN_SEED              = ("train_seed", ControlType.TEXTBOX, None, -1, int, EnableConfig.ALL)
    TRAIN_TEXTENCODER_LEARNING_RATE = ("train_textencoder_learning_rate", ControlType.TEXTBOX, None, "", float, EnableConfig.LORA)
    TRAIN_MODEL_PRECISION   = ("train_model_precision", ControlType.DROPDOWN, PRECISION_TYPES[:3], "fp16", str, EnableConfig.ALL)
    TRAIN_LORA_PRECISION    = ("train_lora_precision", ControlType.DROPDOWN, PRECISION_TYPES[:3], "fp32", str, EnableConfig.ALL)
    IMAGE_BUCKETS_STEP      = ("image_buckets_step", ControlType.DROPDOWN, IMAGESTEPS, 256, int, EnableConfig.LORA)
    IMAGE_MIN_LENGTH        = ("image_min_length", ControlType.TEXTBOX, None, 512, int, EnableConfig.LORA)
    IMAGE_MAX_RATIO         = ("image_max_ratio", ControlType.TEXTBOX, None, 2, float, EnableConfig.LORA)
    SUB_IMAGE_NUM           = ("sub_image_num", ControlType.TEXTBOX, None, 0, int, EnableConfig.LORA)
    IMAGE_MIRRORING         = ("image_mirroring", ControlType.CHECKBOX, None, False, bool, EnableConfig.LORA)
    IMAGE_USE_FILENAME_AS_TAG = ("image_use_filename_as_tag", ControlType.CHECKBOX, None, False, bool, EnableConfig.LORA)
    IMAGE_DISABLE_UPSCALE   = ("image_disable_upscale", ControlType.CHECKBOX, None, False, bool, EnableConfig.LORA)
    SAVE_PER_STEPS          = ("save_per_steps", ControlType.TEXTBOX, None, 0, int, EnableConfig.ALL)
    SAVE_PRECISION          = ("save_precision", ControlType.DROPDOWN, PRECISION_TYPES[:3], "fp16", str, EnableConfig.ALL)
    SAVE_OVERWRITE          = ("save_overwrite", ControlType.CHECKBOX, None, False, bool, EnableConfig.ALL)
    SAVE_AS_JSON            = ("save_as_json", ControlType.CHECKBOX, None, False, bool, EnableConfig.NDIFF2)

    DIFF_SAVE_1ST_PASS      = ("diff_save_1st_pass", ControlType.CHECKBOX, None, False, bool, EnableConfig.DIFF1)
    DIFF_1ST_PASS_ONLY      = ("diff_1st_pass_only", ControlType.CHECKBOX, None, False, bool, EnableConfig.DIFF1)
    DIFF_LOAD_1ST_PASS      = ("diff_load_1st_pass", ControlType.TEXTBOX, None, "", str, EnableConfig.DIFF1)
    TRAIN_LR_STEP_RULES     = ("train_lr_step_rules", ControlType.TEXTBOX, None, "", str, EnableConfig.ALL)
    TRAIN_LR_WARMUP_STEPS   = ("train_lr_warmup_steps", ControlType.TEXTBOX, None, 0, int, EnableConfig.ALL)
    TRAIN_LR_SCHEDULER_NUM_CYCLES = ("train_lr_scheduler_num_cycles", ControlType.TEXTBOX, None, 1, int, EnableConfig.ALL)
    TRAIN_LR_SCHEDULER_POWER = ("train_lr_scheduler_power", ControlType.TEXTBOX, None, 1.0, float, EnableConfig.ALL)
    TRAIN_SNR_GAMMA         = ("train_snr_gamma", ControlType.TEXTBOX, None, 5, float, EnableConfig.ALL)
    TRAIN_FIXED_TIMSTEPS_IN_BATCH = ("train_fixed_timsteps_in_batch", ControlType.CHECKBOX, None, False, bool, EnableConfig.ALL)
    IMAGE_USE_TRANSPARENT_BACKGROUND_AJUST = ("image_use_transparent_background_ajust", ControlType.CHECKBOX, None, False, bool, EnableConfig.ALL)

    LOGGING_VERBOSE         = ("logging_verbose", ControlType.CHECKBOX, None, False, bool, EnableConfig.NDIFF2)
    LOGGING_SAVE_CSV        = ("logging_save_csv", ControlType.CHECKBOX, None, False, bool, EnableConfig.NDIFF2)
    MODEL_V_PRED            = ("model_v_pred", ControlType.CHECKBOX, None, False, bool, EnableConfig.ALL)

    NETWORK_BLOCKS          = ("network_blocks(BASE = TextEncoder)", ControlType.CHECKBOXGROUP, BLOCKID26, BLOCKID26, list, EnableConfig.ALL)

    #unuased parameters
    # LOGGING_USE_WANDB       = ("logging_use_wandb", ControlType.CHECKBOX, None, False, bool)
    # TRAIN_REPEAT            = ("train_repeat", ControlType.TEXTBOX, None, 1.0, int, EnableConfig.ALL)
    # TRAIN_USE_BUCKET        = ("train_use_bucket", ControlType.CHECKBOX, None, False, bool)
    # TRAIN_OPTIMIZER_ARGS    = ("train_optimizer_args", ControlType.TEXTBOX, None, "", str)
    # LOGGING_DIR             = ("logging_dir", ControlType.TEXTBOX, None, "", str, EnableConfig.NDIFF2)
    # GRADIENT_ACCUMULATION_STEPS = ("gradient_accumulation_steps", ControlType.TEXTBOX, None, "1", str, EnableConfig.ALL)
    # GEN_NOISE_SCHEDULER     = ("gen_noise_scheduler", ControlType.DROPDOWN, NOISE_SCHEDULERS, NOISE_SCHEDULERS[6], str, EnableConfig.NDIFF2)
    # LORA_TRAIN_TARGETS      = ("lora_train_targets", ControlType.RADIO, TARGET_MODULES, TARGET_MODULES[0], str, EnableConfig.LORA)
    # LOGGING_USE_TENSORBOARD = ("logging_use_tensorboard", ControlType.CHECKBOX, False, "", bool, EnableConfig.NDIFF2)
    # TRAIN_MIN_TIMESTEPS     = ("train_min_timesteps", ControlType.TEXTBOX, None, 0, int, EnableConfig.ALL)
    # TRAIN_MAX_TIMESTEPS     = ("train_max_timesteps", ControlType.TEXTBOX, None, 1000, int, EnableConfig.ALL)

    def __init__[T](self, name: str, control_type: ControlType, choices: typing.Sequence[T] | None, default: T, value_type: type[T], enable: EnableConfig):
        self.NAME = name
        self.CONTROL_TYPE = control_type
        self.CHOICES = choices
        self.DEFAULT = default
        self.VALUE_TYPE = value_type
        self.ENABLE = enable

@dataclass(slots=True)
class ComponentConfig:
    config: ControlConfig
    instance: Component = None

@dataclass(slots=True)
class ComponentValue(ComponentConfig):
    value: typing.Any = None
    update: numbers.Number | str | dict[str, typing.Any] | None = None

@dataclass(init=False, slots=True)
class ConfigBase[T]:
    network_type: T
    network_rank: T
    network_alpha: T
    lora_data_directory: T
    lora_trigger_word: T
    image_size: T
    train_iterations: T
    train_batch_size: T
    train_learning_rate: T
    train_optimizer: T
    train_lr_scheduler: T
    save_lora_name: T
    use_gradient_checkpointing: T
    network_blocks: T
    network_conv_rank: T
    network_conv_alpha: T
    network_element: T
    image_buckets_step: T
    image_min_length: T
    image_max_ratio: T
    sub_image_num: T
    image_mirroring: T
    image_use_filename_as_tag: T
    image_disable_upscale: T
    image_use_transparent_background_ajust: T
    train_fixed_timsteps_in_batch: T
    train_textencoder_learning_rate: T
    train_seed: T
    train_lr_step_rules: T
    train_lr_warmup_steps: T
    train_lr_scheduler_num_cycles: T
    train_lr_scheduler_power: T
    train_snr_gamma: T
    save_per_steps: T
    train_model_precision: T
    train_lora_precision: T
    save_precision: T
    diff_load_1st_pass: T
    diff_save_1st_pass: T
    diff_1st_pass_only: T
    logging_save_csv: T
    logging_verbose: T
    save_overwrite: T
    save_as_json: T
    model_v_pred: T

    _ALL_KEYS: list[str]
    """
    すべての要素のキーを保持します.

    Note:
        __slots__が継承関係を追えないため、全てのキーを保持するための変数です.
    """

    def __init__(self, type: T):
        super(ConfigBase, self).__init__()
        self.network_type = type(ControlConfig.NETWORK_TYPE)
        self.network_rank = type(ControlConfig.NETWORK_RANK)
        self.network_alpha = type(ControlConfig.NETWORK_ALPHA)
        self.lora_data_directory = type(ControlConfig.LORA_DATA_DIRECTORY)
        self.lora_trigger_word = type(ControlConfig.LORA_TRIGGER_WORD)
        self.image_size = type(ControlConfig.IMAGE_SIZE)
        self.train_iterations = type(ControlConfig.TRAIN_ITERATIONS)
        self.train_batch_size = type(ControlConfig.TRAIN_BATCH_SIZE)
        self.train_learning_rate = type(ControlConfig.TRAIN_LEARNING_RATE)
        self.train_optimizer = type(ControlConfig.TRAIN_OPTIMIZER)
        self.train_lr_scheduler = type(ControlConfig.TRAIN_LR_SCHEDULER)
        self.save_lora_name = type(ControlConfig.SAVE_LORA_NAME)
        self.use_gradient_checkpointing = type(ControlConfig.USE_GRADIENT_CHECKPOINTING)
        self.network_blocks = type(ControlConfig.NETWORK_BLOCKS)
        self.network_conv_rank = type(ControlConfig.NETWORK_CONV_RANK)
        self.network_conv_alpha = type(ControlConfig.NETWORK_CONV_ALPHA)
        self.network_element = type(ControlConfig.NETWORK_ELEMENT)
        self.image_buckets_step = type(ControlConfig.IMAGE_BUCKETS_STEP)
        self.image_min_length = type(ControlConfig.IMAGE_MIN_LENGTH)
        self.image_max_ratio = type(ControlConfig.IMAGE_MAX_RATIO)
        self.sub_image_num = type(ControlConfig.SUB_IMAGE_NUM)
        self.image_mirroring = type(ControlConfig.IMAGE_MIRRORING)
        self.image_use_filename_as_tag = type(ControlConfig.IMAGE_USE_FILENAME_AS_TAG)
        self.image_disable_upscale = type(ControlConfig.IMAGE_DISABLE_UPSCALE)
        self.image_use_transparent_background_ajust = type(ControlConfig.IMAGE_USE_TRANSPARENT_BACKGROUND_AJUST)
        self.train_fixed_timsteps_in_batch = type(ControlConfig.TRAIN_FIXED_TIMSTEPS_IN_BATCH)
        self.train_textencoder_learning_rate = type(ControlConfig.TRAIN_TEXTENCODER_LEARNING_RATE)
        self.train_seed = type(ControlConfig.TRAIN_SEED)
        self.train_lr_step_rules = type(ControlConfig.TRAIN_LR_STEP_RULES)
        self.train_lr_warmup_steps = type(ControlConfig.TRAIN_LR_WARMUP_STEPS)
        self.train_lr_scheduler_num_cycles = type(ControlConfig.TRAIN_LR_SCHEDULER_NUM_CYCLES)
        self.train_lr_scheduler_power = type(ControlConfig.TRAIN_LR_SCHEDULER_POWER)
        self.train_snr_gamma = type(ControlConfig.TRAIN_SNR_GAMMA)
        self.save_per_steps = type(ControlConfig.SAVE_PER_STEPS)
        self.train_model_precision = type(ControlConfig.TRAIN_MODEL_PRECISION)
        self.train_lora_precision = type(ControlConfig.TRAIN_LORA_PRECISION)
        self.save_precision = type(ControlConfig.SAVE_PRECISION)
        self.diff_load_1st_pass = type(ControlConfig.DIFF_LOAD_1ST_PASS)
        self.diff_save_1st_pass = type(ControlConfig.DIFF_SAVE_1ST_PASS)
        self.diff_1st_pass_only = type(ControlConfig.DIFF_1ST_PASS_ONLY)
        self.logging_save_csv = type(ControlConfig.LOGGING_SAVE_CSV)
        self.logging_verbose = type(ControlConfig.LOGGING_VERBOSE)
        self.save_overwrite = type(ControlConfig.SAVE_OVERWRITE)
        self.save_as_json = type(ControlConfig.SAVE_AS_JSON)
        self.model_v_pred = type(ControlConfig.MODEL_V_PRED)
        self._ALL_KEYS = list(self.__slots__)

    def components[T](self, type: T=ComponentConfig) -> typing.Iterator[T]:
        """
        設定値のイテレーターを返します.

        Args:
            type (T, optional): 列挙する型. ComponentConfigの派生型である必要があります. デフォルトでは ComponentConfig.

        Yields:
            typing.Iterator[T]: 指定された型のイテレーター.
        """
        for key in self._ALL_KEYS:
            value = getattr(self, key)
            if isinstance(value, type):
                yield value

    def copy(self, dest: 'ConfigBase[ComponentConfig]') -> None:
        for key in self._ALL_KEYS:
            src_value = getattr(self, key)
            dest_value = getattr(dest, key)
            if (isinstance(src_value, ComponentConfig)) and (isinstance(dest_value, ComponentConfig)):
                dest_value.instance = src_value.instance
            else:
                logger.debug(f"[TrainTrain] Unexpected variable type in copy: {key}")


@dataclass(init=False, slots=True)
class ConfigRoot[T](ConfigBase[T]):
    mode: T
    model: T
    model_type: T
    vae: T
    original_prompt: T
    target_prompt: T
    use_2nd_pass_settings: T
    second_pass: ConfigBase[T] # "2nd pass"

    _COMPORNENTS: set[Component] | None = None

    @typing.override
    def __init__(self, type: T):
        # NOTE: Python 3.12.8 では super() が使えない
        # BUG: https://github.com/python/cpython/issues/90562
        super(ConfigRoot, self).__init__(type)
        self.mode = type(ControlConfig.MODE)
        self.model = type(ControlConfig.MODEL)
        self.model_type = type(ControlConfig.MODEL_TYPE)
        self.vae = type(ControlConfig.VAE)
        self.original_prompt = type(ControlConfig.ORIGINAL_PROMPT)
        self.target_prompt = type(ControlConfig.TARGET_PROMPT)
        self.use_2nd_pass_settings = type(ControlConfig.USE_2ND_PASS_SETTINGS)
        self.second_pass = ConfigBase[T](type)

        # 内部用
        self._ALL_KEYS = list(super(ConfigRoot, self).__slots__) + list(self.__slots__)
        self._COMPORNENTS = None

    @typing.override
    def copy(self, dest: 'ConfigRoot[ComponentConfig]') -> None:
        super(ConfigRoot, self).copy(dest)
        # 2nd pass分もコピー
        self.second_pass.copy(dest.second_pass)
        # コンポーネントをコピー
        dest._COMPORNENTS = self._COMPORNENTS

def get_component[T](config: ConfigBase[T], name: str, type: T=ComponentConfig) -> T:
    for component in config.components(type):
        if component.config.NAME == name:
            return component
    raise KeyError(f"ControlConfig {name} not found")

def get_value(config: ConfigBase[ComponentValue], name: str) -> typing.Any:
    for component in config.components(ComponentValue):
        if component.config.NAME == name:
            return component.value
    raise KeyError(f"ControlConfig {name} not found")

def get_instance(config: ConfigBase[ComponentConfig], target: ControlConfig) -> Component:
    for component in config.components(ComponentConfig):
        if component.config == target:
            return component.instance
    raise KeyError(f"ControlConfig {config} not found")

def _create_control(value: ComponentConfig, pas: int = 0) -> Component:
    instance = None
    label = value.config.NAME.replace("_"," ")
    visible = value.config.ENABLE[0] if pas != 2 else value.config.ENABLE[3]
    match (value.config.CONTROL_TYPE):
        case ControlType.DROPDOWN:
            if value.config.CHOICES is None:
                default = None
            elif not value.config.DEFAULT is None:
                default = value.config.DEFAULT
            else:
                default = value.config.CHOICES[0]
            instance = gr.Dropdown(
                elem_id=f"tt{pas}:{value.config.NAME}",
                label=label,
                choices=value.config.CHOICES,
                value=default,
                visible=visible,
                interactive=True,
                render=False)
        case ControlType.TEXTBOX:
            instance = gr.Textbox(
                elem_id=f"tt{pas}:{value.config.NAME}",
                label=label,
                value=value.config.DEFAULT,
                visible=visible,
                interactive=True,
                render=False)
        case ControlType.TEXTAREA:
            instance = gr.TextArea(
                elem_id=f"tt{pas}:{value.config.NAME}",
                label=label,
                value=value.config.DEFAULT,
                visible=visible,
                lines=3,
                interactive=True,
                render=False)
        case ControlType.CHECKBOX:
            instance = gr.Checkbox(
                elem_id=f"tt{pas}:{value.config.NAME}",
                label=label,
                value=value.config.DEFAULT,
                visible=visible,
                interactive=True,
                render=False)
        case ControlType.CHECKBOXGROUP:
            instance = gr.CheckboxGroup(
                elem_id=f"tt{pas}:{value.config.NAME}",
                label=label,
                choices=value.config.CHOICES,
                value=value.config.DEFAULT,
                type="value",
                visible=visible,
                interactive=True,
                render=False)
        case ControlType.RADIO:
            instance = gr.Radio(
                elem_id=f"tt{pas}:{value.config.NAME}",
                label=label,
                choices=value.config.CHOICES, # choices=[x + " " for x in i.CHOICES] if pas > 0 else i.CHOICES
                value=value.config.DEFAULT,
                visible=visible,
                interactive=True,
                render=False)
        case _:
            raise ValueError(f"ControlType {value.config.CONTROL_TYPE} was illegal type")

    value.instance = instance
    return instance

def _create(config: ConfigBase[ComponentConfig], pas: int = 0) -> list[Component]:
    result = []
    for component in config.components(ComponentConfig):
        result.append(_create_control(component, pas))
    return result

def create(config: ConfigRoot[ComponentConfig]) -> None:
    # コントロールを作成
    list_components = _create(config, 1) + _create(config.second_pass, 2)

    # イベントハンドラー用にコンポーネントを保持
    config._COMPORNENTS = set(list_components)

def _concat_blocksets(base: set[Block], append: Block | typing.Sequence[Block] | set[Block] | None) -> set[Block]:
    result = base.copy()
    if append is None:
        pass
    elif isinstance(append, Block):
        result.add(append)
    else:
        result.update(append)
    return result

def _input_convert(config: ConfigRoot[ComponentConfig],
                   inputs: Block | typing.Sequence[Block] | set[Block] | None,
                   args: dict) -> tuple[ConfigRoot[ComponentValue], typing.Any]:
    # 返却用のコピーオブジェクトを作成
    values = ConfigRoot(ComponentValue)
    config.copy(values)

    # 入力マッピング
    def mapping(values: ConfigRoot[ComponentValue]) -> None:
        for component in values.components(ComponentValue):
            component.value = args[component.instance]

    mapping(values)
    mapping(values.second_pass)

    # 追加の入力をマッピング
    append_args = None
    if inputs is None:
        pass
    elif isinstance(inputs, Block):
        append_args = args[inputs]
    elif isinstance(inputs, set):
        append_args = {}
        for component in inputs:
            if not isinstance(component, Block):
                raise TypeError(f"Input type {type(component)} was illegal type")
            append_args[component] = args[component]
    elif isinstance(inputs, typing.Sequence):
        append_args = []
        for component in inputs:
            if not isinstance(component, Block):
                raise TypeError(f"Input type {type(component)} was illegal type")
            append_args.append(args[component])
    else:
        raise TypeError(f"Input type {type(inputs)} was illegal type")

    return values, append_args

def _output_convert(values: ConfigRoot[ComponentValue],
                    outputs: Block | typing.Sequence[Block] | set[Block] | None,
                    result: typing.Any | None) -> dict:
    updates = {}

    # 出力マッピング
    def mapping(values: ConfigRoot[ComponentValue]) -> None:
        for component in values.components(ComponentValue):
            if not component.update is None:
                updates[component.instance] = component.update

    mapping(values)
    mapping(values.second_pass)

    # 通常の関数戻り値を追加する
    if outputs is None:
        pass
    elif isinstance(outputs, Block):
        updates[outputs] = result
    elif isinstance(result, dict):
        for key, value in result.items():
            if not isinstance(key, Block):
                raise TypeError(f"Return type {type(key)} was illegal type")
            else:
                updates[key] = value
    elif isinstance(outputs, typing.Sequence):
        if len(outputs) != len(result):
            raise IndexError(f"Return length {len(result)} and output length {len(outputs)} was not matched")
        for i in range(len(outputs)):
            updates[outputs[i]] = result[i]
    else:
        raise TypeError(f"Output type {type(result)} was illegal type")

    return updates

def _converter(config: ConfigRoot[ComponentConfig],
               callback: typing.Callable[[ConfigRoot[ComponentValue], typing.Any], typing.Any | None],
               inputs: Block | typing.Sequence[Block] | set[Block] | None,
               outputs: Block | typing.Sequence[Block] | set[Block] | None,
               args: dict
               ) -> dict:
    values, append = _input_convert(config, inputs, args)
    return _output_convert(values, outputs, callback(values, append))

class EventHandler(typing.Protocol):
    def __call__(self,
                 fn: typing.Callable[..., typing.Any] | None = None,
                 inputs: Block | typing.Sequence[Block] | set[Block] | None = None,
                 outputs: Block | typing.Sequence[Block] | None = None,
                 **kwargs: typing.Any) -> typing.Any:
        ...

def event_proxy(config: ConfigRoot[ComponentConfig],
                event: EventHandler,
                callback: typing.Callable[[ConfigRoot[ComponentValue], typing.Any], typing.Any | None],
                inputs: Block | typing.Sequence[Block] | set[Block] | None = None,
                outputs: Block | typing.Sequence[Block] | set[Block] | None = None,
                **kwargs: typing.Any) -> None:

    event(fn=lambda args: _converter(config, callback, inputs, outputs, args),
          inputs=_concat_blocksets(config._COMPORNENTS, inputs),
          outputs=_concat_blocksets(config._COMPORNENTS, outputs),
          **kwargs)

def as_dict(config: ConfigRoot[ComponentValue]) -> dict[str, typing.Any]:
    result = {}
    # 1st pass
    for component in config.components(ComponentValue):
        result[component.config.NAME] = component.value
    # 2nd pass
    result["2nd pass"] = {}
    for component in config.second_pass.components(ComponentValue):
        result["2nd pass"][component.config.NAME] = component.value
    return result

def export_json(dir: str, values: dict[str, typing.Any], timestamp: bool = False) -> None:
    current_time = datetime.now()
    export = copy.deepcopy(values)

    # 2nd passを付与するかどうか
    if (export["mode"] != "Difference") and ("2nd pass" in export):
        del export["2nd pass"]

    # JSONファイル出力
    add = "" if not timestamp else f"-{current_time.strftime('%Y%m%d_%H%M%S')}"
    filepath = os.path.join(dir, f"{export['save_lora_name']}{add}.json")
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(filepath, "w", encoding='utf-8') as file:
        json.dump(export, file, indent=4)

def apply_dict(config: ConfigRoot[ComponentValue], values: dict[str, typing.Any]) -> None:
    def find_value_dropdown(value: typing.Any, choices: typing.Any) -> bool:
        result = False
        if (isinstance(choices, list)) and (len(choices) > 0):
            if isinstance(choices[0], tuple):
                # キーと値の組み合わせ
                result = (value in [x[1] for x in choices])
            else:
                # 値のみ
                result = (value in choices)
        return result

    def applier(component: ComponentValue, dic: dict[str, typing.Any]) -> None:
        if component.config.NAME in dic:
            value = dic[component.config.NAME]
            if not isinstance(value, component.config.VALUE_TYPE):
                try:
                    value = component.config.VALUE_TYPE(value)
                except ValueError:
                    logger.warning(f"[TrainTrain] Ignored value: Illegal value type. {component.config.NAME}={dic[component.config.NAME]}")
                    value = component.config.DEFAULT
            if component.config.CONTROL_TYPE == ControlType.DROPDOWN:
                if (component.instance.choices is None) or (len(component.instance.choices) == 0):
                    logger.warning(f"[TrainTrain] Ignored value: The list of choices was Empty. {component.config.NAME}")
                    value = None
                elif not find_value_dropdown(value, component.instance.choices):
                    logger.warning(f"[TrainTrain] Ignored value: Not in the list of choices. {component.config.NAME}={value}")
                    value = component.config.DEFAULT if not component.config.DEFAULT is None else component.instance.choices[0]
        else:
            value = component.config.DEFAULT

        component.value = value
        component.update = value

    # 1st pass
    for component in config.components(ComponentValue):
        applier(component, values)
    # 2nd pass
    if "2nd pass" in values:
        for component in config.second_pass.components(ComponentValue):
            applier(component, values["2nd pass"])

def import_json(filepath: str) -> dict[str, typing.Any]:
    with open(filepath, "r", encoding='utf-8') as file:
        return json.load(file)

def get_queue_header(config: ConfigRoot[ComponentConfig]) -> list[str]:
    # 先頭はキュー名(現在はsave_lora_nameと同じ)
    result: list[str] = ["name"]
    # 1st pass
    for component in config.components():
        result.append(component.config.NAME)
    # 2nd pass
    for component in config.second_pass.components():
        result.append(f"2nd pass:{component.config.NAME}")
    return result

def list_queues(config: ConfigRoot[ComponentConfig], queue_list: dict[str, dict]) -> list[list[str]]:
    # 項目名を作成する
    first_keys = [component.config.NAME for component in config.components()]
    second_keys = [component.config.NAME for component in config.second_pass.components()]

    lists: list[list[str]] = []
    for name, item in queue_list.items():
        # 先頭はキュー名
        values = [name]
        # 1st pass
        values += [item.get(key, "") for key in first_keys]
        # 2nd pass
        values += [item.get(key, "") for key in second_keys]
        lists.append(values)
    return lists
