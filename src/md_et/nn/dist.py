import os
from contextlib import nullcontext
from datetime import timedelta
from typing import Literal

import torch as th
import torch.distributed as dist
from loguru import logger


def cleanup_dist() -> None:
    dist.destroy_process_group()


def setup_dist(rank, world_size, port="33281") -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    # initialize the process group
    dist_type = "nccl" if th.cuda.is_available() else "gloo"
    if th.cuda.is_available():
        th.cuda.set_device(rank)
    dist.init_process_group(
        dist_type,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=10),
        init_method="env://",
    )


def setup_device(rank=0) -> th.device:
    th.backends.cuda.matmul.allow_tf32 = True
    th.backends.cudnn.allow_tf32 = True

    if th.cuda.is_available():
        th.cuda.set_device(rank)
        device = th.device("cuda", rank)
    else:
        device = th.device("cpu")

    logger.info(f"Rank {rank} running, accelerator: {device}")

    return device


ptd_name_to_type = {
    "float32": th.float32,
    "bfloat16": th.bfloat16,
    "float16": th.float16,
    "float64": th.float64,
}


def get_amp(
    ptdtype: Literal["float32", "bfloat16", "float16", "float64"] | None = None,
) -> nullcontext | th.amp.autocast:
    if ptdtype == "bfloat16" and not th.cuda.is_bf16_supported():
        raise ValueError("BF16 is not supported on this device")
    ptdtype = th.get_default_dtype() if ptdtype is None else ptd_name_to_type[ptdtype]
    logger.info(f"Casting to type: {ptdtype}")
    default_float_dtype = th.get_default_dtype()
    assert not (default_float_dtype == th.float64 and ptdtype != th.float64), (
        "Cannot use autocast with double precision"
    )

    ctx = (
        th.amp.autocast(device_type="cuda", dtype=ptdtype)
        if (th.cuda.is_available() and ptdtype != th.float64)
        else nullcontext()
    )
    return ctx


def set_dtype(fp_type: th.dtype | None = None) -> th.dtype:
    if fp_type is None:
        return th.get_default_dtype()
    if fp_type != th.get_default_dtype():
        logger.warning(
            f"Using {fp_type} precision, deviating from default precision {th.get_default_dtype()}. This is probably unintended."
        )
    return fp_type


def assert_dtype(tensor: th.Tensor, dtype: th.dtype) -> None:
    assert tensor.dtype == dtype, f"Tensor should be of type {dtype}, but is {tensor.dtype}"
