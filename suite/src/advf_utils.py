import torch


def get_module_zeroer(device, adapter_name):
    def set_module_zero(_, module):
        if hasattr(module, "adapters") and adapter_name in module.adapters:
            new_down = torch.nn.Linear(768, 48).to(device)
            new_down.weight.data.fill_(0)
            new_down.bias.data.fill_(0)
            new_down.weight.requires_grad = False
            new_down.bias.requires_grad = False
            module.adapters[adapter_name].adapter_down[0] = new_down
            new_up = torch.nn.Linear(48, 768).to(device)
            new_up.weight.data.fill_(0)
            new_up.bias.data.fill_(0)
            new_up.weight.requires_grad = False
            new_up.bias.requires_grad = False
            module.adapters[adapter_name].adapter_up = new_up

    return set_module_zero

def get_module_freezer(adapter_name):
    def set_module_zero(_, module):
        if hasattr(module, "adapters") and adapter_name in module.adapters:
            module.adapters[adapter_name].adapter_down[0].weight.requires_grad = False
            module.adapters[adapter_name].adapter_down[0].bias.requires_grad = False
            module.adapters[adapter_name].adapter_up.weight.requires_grad = False
            module.adapters[adapter_name].adapter_up.bias.requires_grad = False
    return set_module_zero

def zero_freeze_adapter(model, adapter_name, model_dtype):
    set_module_zero = get_module_zeroer(model.device, adapter_name)
    model.apply_to_adapter_layers(set_module_zero)
    model.adapter_to(adapter_name, device=model.device, dtype=model_dtype)

def freeze_adapter(model, adapter_name):
    freeze_module = get_module_freezer(adapter_name)
    model.apply_to_adapter_layers(freeze_module)


def unfreeze_reload_adapter(model, path, name):
    model.load_adapter(
        path,
        load_as=name,
        set_active=True,
    )
