import os
from logging_utils import logger
import peft


def get_peft_config(adapter_config):
    match adapter_config:
        case "lora":
            return peft.tuners.lora.LoraConfig(
                r=64,
                lora_alpha=32,
                lora_dropout=0.1,
                # bias="none",
                target_modules=["q_proj", "k_proj", "v_proj"],
                task_type="CAUSAL_LM",
            )

        case "ia3":
            return peft.tuners.ia3.IA3Config(
                # r=64,
                # lora_alpha=32,
                # lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "down_proj"],
                feedforward_modules=["down_proj"],
                task_type="CAUSAL_LM",
            )

        case _:
            return None


def init_peft_adapter(adapter_config, config_title, model):
    peft_config = get_peft_config(adapter_config)
    if peft_config is not None:
        adapter_name = f"{config_title}_adapter"
        logger.info(f"Setting a new PEFT titled {adapter_name}")
        model = peft.mapping.get_peft_model(
            model,
            peft_config,
            adapter_name=adapter_name,
            autocast_adapter_dtype=False,
        )
    else:
        logger.warning(f"Failed to init peft adapter: Invalid PEFT config: {adapter_config}")
        raise ValueError(f"Invalid PEFT config: {adapter_config}")


def init_and_load_peft_adapter(adapter_path, config_title, model, device=None):
    if adapter_path and os.path.isdir(adapter_path):
        adapter_name = f"{config_title}_adapter"
        logger.info(f"Loading PEFT from {adapter_path}")
        peft.peft_model.PeftModel.from_pretrained(
            model,
            adapter_path,
            adapter_name=adapter_name,
            is_trainable=True,
            torch_device=device,
        )
    else:
        logger.warning(f"Failed to load peft adapter: Invalid PEFT path: {adapter_path}")
        raise ValueError(f"Invalid PEFT path: {adapter_path}")
