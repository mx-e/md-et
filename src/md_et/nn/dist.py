import torch as th
from loguru import logger


def set_dtype(fp_type: th.dtype | None = None) -> th.dtype:
    if fp_type is None:
        return th.get_default_dtype()
    if fp_type != th.get_default_dtype():
        logger.warning(
            f"Using {fp_type} precision, deviating from default precision {th.get_default_dtype()}. This is probably unintended."
        )
    return fp_type
