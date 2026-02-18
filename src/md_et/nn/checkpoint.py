import torch as th
from loguru import logger


def load_checkpoint(model, path, dtype: th.dtype | None = None) -> None:
    device = next(model.parameters()).device
    checkpoint = th.load(path, weights_only=True, map_location=device)

    if dtype is None:
        dtype = th.get_default_dtype()

    model_state = checkpoint["model_state_dict"]
    first_tensor = next((t for t in model_state.values() if isinstance(t, th.Tensor)), None)
    if first_tensor is not None and first_tensor.dtype != dtype:
        logger.warning(f"Converting checkpoint weights to {dtype}")
        for key, tensor in model_state.items():
            if isinstance(tensor, th.Tensor) and tensor.dtype.is_floating_point:
                model_state[key] = tensor.to(dtype)

    if hasattr(model, "module"):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)
