import torch as th
from loguru import logger


def save_checkpoint(model, optimizer, step, path, ema=None) -> None:
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }
    if ema is not None:
        ckpt["ema_state"] = ema.state_dict()
    th.save(ckpt, path)


def load_checkpoint(  # noqa: C901
    model, path, optimizer=None, ema=None, step_back=200, dtype: th.dtype | None = None
) -> int:
    device = next(model.parameters()).device
    checkpoint = th.load(path, weights_only=True, map_location=device)

    # Convert state dict to specified dtype if requested
    if dtype is None:
        dtype = th.get_default_dtype()
        if dtype != th.get_default_dtype():
            logger.warning(
                f"Using {dtype} precision for loading checkpoint, deviating from default precision {th.get_default_dtype()}. This is probably unintended."
            )
        # Check if conversion is necessary by examining the first tensor's dtype
        model_state = checkpoint["model_state_dict"]
        first_tensor = next((t for t in model_state.values() if isinstance(t, th.Tensor)), None)
        if first_tensor is not None and first_tensor.dtype != dtype:
            logger.warning(f"Converting checkpoint weights to {dtype}")
            # Conversion is necessary
            for key, tensor in model_state.items():
                if isinstance(tensor, th.Tensor) and tensor.dtype.is_floating_point:
                    model_state[key] = tensor.to(dtype)

            if "ema_state" in checkpoint and ema is not None:
                ema_state = checkpoint["ema_state"]
                for key, tensor in ema_state.items():
                    if isinstance(tensor, th.Tensor) and tensor.dtype.is_floating_point:
                        ema_state[key] = tensor.to(dtype)

    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    if ema is not None and "ema_state" in checkpoint:
        ema.load_state_dict(checkpoint["ema_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["step"] - step_back
