import os
from src.general_utits import ensure_path_exists
from src.logging_utils import logger
import adapters
from src.advf_utils import freeze_adapter


def get_ah_config(adapter_config):
    match adapter_config:
        case "compacter":
            return adapters.CompacterConfig(
                # phm_dim=64,
                # phm_rank=32,
                # mh_adapter=True,
                # output_adapter=True,
            )

        case "ia3":
            return adapters.IA3Config()

        case "lora":
            return adapters.LoRAConfig(
                # r=16,
                alpha=16,
                dropout=0.05,
                attn_matrices=["q", "k", "v"],
            )

        case "pfeiffer":
            return adapters.SeqBnConfig()

        case _:
            return None


def init_ah_advfusion(
    advadp_path_list,
    advfusion_target,
    model,
    model_dtype,
):
    target_adapter_path = None
    target_adapter_name = None

    adapters.init(model)
    fusion_adapter_names = []
    for path in advadp_path_list:
        name = path.split("-")[-1]
        if os.path.isdir(path):
            fusion_adapter_names.append(name)
            logger.info(f"Loading frozen adapter: {name}")
            model.load_adapter(path, load_as=name, set_active=True)
            freeze_adapter(model, name)
            if advfusion_target in path:
                logger.info(f"Zeroing adapter: {name}")
                target_adapter_path = path
                target_adapter_name = name
            model.adapter_to(name, device=model.device, dtype=model_dtype)
        else:
            logger.info(f"Invalid adapter path: {path}")

    fusion_name = adapters.composition.Fuse(*fusion_adapter_names)
    model.add_adapter_fusion(fusion_name, set_active=True)
    model.adapter_fusion_to(fusion_name, device=model.device, dtype=model_dtype)
    model.train_adapter_fusion(fusion_name)

    return target_adapter_path, target_adapter_name, fusion_name


def init_ah_adapter(adapter_config, config_title, model, model_dtype):
    adapters.init(model)
    adapter_name = f"{config_title}_adapter"
    config = get_ah_config(adapter_config)
    model.add_adapter(adapter_name, config=config)
    model.adapter_to(adapter_name, device=model.device, dtype=model_dtype)
    model.train_adapter(adapter_name)

    return adapter_name


def load_ah_adapter(adapter_path, adapter_name, model, set_active=True):
    if adapter_path and os.path.isdir(adapter_path):
        model.load_adapter(
            adapter_path,
            load_as=adapter_name,
            set_active=set_active,
        )


def save_ah_adapter(adapter_path, adapter_config, adapter_name, model):
    ensure_path_exists(adapter_path)
    if adapter_config == "advfusion":
        model.save_adapter_fusion(adapter_path, adapter_name)
        logger.info(f"Fusion {adapter_name} saved to {adapter_path}")
    else:
        model.save_adapter(adapter_path, adapter_name)
        logger.info(f"Adapter {adapter_name} saved to {adapter_path}")
