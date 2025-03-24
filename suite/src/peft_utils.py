import os
from src.logging_utils import logger
import peft
import torch


def get_peft_config(adapter_config, is_decoder_only=False):
    match adapter_config:
        case "lora":
            return peft.LoraConfig(
                # r=16,
                lora_alpha=16,
                lora_dropout=0.05,
                # bias="none",
                target_modules=["q_proj", "k_proj", "v_proj"]
                if is_decoder_only
                else ["q", "k", "v"],
                task_type="CAUSAL_LM" if is_decoder_only else "SEQ_2_SEQ_LM",
            )

        case "ia3":
            return peft.IA3Config(
                target_modules=["q_proj", "k_proj", "v_proj"] #"down_proj"
                if is_decoder_only
                else ["q", "k", "v"], #"o"
                # feedforward_modules=["q_proj", "k_proj", "v_proj", "down_proj"]
                # if is_decoder_only
                # else ["q", "k", "v", "o"],
                task_type="CAUSAL_LM" if is_decoder_only else "SEQ_2_SEQ_LM",
            )

        case _:
            return None


def init_peft_adapter(adapter_config, config_title, model, is_decoder_only=False):
    peft_config = get_peft_config(adapter_config, is_decoder_only=is_decoder_only)
    if peft_config is not None:
        adapter_name = f"{config_title}_adapter"
        logger.info(f"Setting a new PEFT titled {adapter_name}")
        device = model.device
        model = peft.get_peft_model(
            model,
            peft_config,
            adapter_name=adapter_name,
            # autocast_adapter_dtype=False,
        )
        # peft.cast_mixed_precision_params(model, torch.bfloat16)
        model = model.to(device)

        # # print all model paramter dtypes
        # for name, param in model.named_parameters():
        #     if param.dtype == torch.float32:
        #         logger.info(f"{name}: {param.dtype}")
        #     if "self_attn" in name and param.dtype == torch.float32:
        #         param.data = param.data.to(torch.bfloat16)
        return model
    else:
        logger.warning(
            f"Failed to init peft adapter: Invalid PEFT config: {adapter_config}"
        )
        raise ValueError(f"Invalid PEFT config: {adapter_config}")


def init_and_load_peft_adapter(adapter_path, config_title, model, device=None):
    if adapter_path and os.path.isdir(adapter_path):
        adapter_name = f"{config_title}_adapter"
        logger.info(f"Loading PEFT from {adapter_path}")
        return peft.PeftModel.from_pretrained(
            model,
            adapter_path,
            adapter_name=adapter_name,
            is_trainable=True,
            torch_device=device,
        )
    else:
        logger.warning(
            f"Failed to load peft adapter: Invalid PEFT path: {adapter_path}"
        )
        raise ValueError(f"Invalid PEFT path: {adapter_path}")
