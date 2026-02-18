from functools import wraps

import torch as th
from md_et.nn.types import Property as Props
from md_et.nn.types import property_dtype


def _apply_molwise_func(batch, func, new_props, **kwargs) -> tuple[dict, list]:
    for _, sample in enumerate(batch):
        sample.update(func(sample, **kwargs))
    return batch, new_props


def apply_molwise(new_props) -> callable:
    def create_wrapper(func) -> callable:
        @wraps(func)
        def wrapper(batch, **kwargs) -> tuple[dict, list]:
            return _apply_molwise_func(batch, func, new_props, **kwargs)

        wrapper.__module__ = __name__
        return wrapper

    create_wrapper.__module__ = __name__
    return create_wrapper


@apply_molwise(new_props=[])
def center_positions_on_centroid(mol) -> dict:
    positions = mol[Props.positions]  # (n_atoms, 3)
    centroid = (positions).mean(dim=0, keepdim=True)
    new_positions = positions - centroid
    return {Props.positions: new_positions}


@apply_molwise(new_props=[Props.charge])
def add_default_charge(_) -> dict:
    return {Props.charge: th.tensor([0], dtype=property_dtype(Props.charge))}


@apply_molwise(new_props=[Props.multiplicity])
def add_default_multiplicity(_) -> dict:
    return {
        Props.multiplicity: th.tensor([1], dtype=property_dtype(Props.multiplicity))
    }
