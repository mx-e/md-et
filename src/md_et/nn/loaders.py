import torch as th
from md_et.nn.types import (
    PropertyType as PropsType,
    Property as Props,
    property_dims,
    property_dtype,
    property_type,
)


def batch_tall(batch, props: list[Props], n_atoms) -> dict:
    max_atoms = th.max(n_atoms).item()
    out = {Props.mask: th.zeros(len(batch), max_atoms, dtype=bool)}

    for prop in props:
        if property_type[prop] == PropsType.mol_wise:
            out[prop] = th.stack([sample[prop] for sample in batch])
        elif property_type[prop] == PropsType.atom_wise:
            out[prop] = th.zeros(len(batch), max_atoms, property_dims[prop]).squeeze(-1)
        else:
            raise NotImplementedError(
                f"Props type {property_type[prop]} not supported for tall batching"
            )

    for prop in props:
        if property_type[prop] == PropsType.atom_wise:
            for i, sample in enumerate(batch):
                out[prop][i, : n_atoms[i]] = sample[prop]
                out[Props.mask][i, : n_atoms[i]] = 1
    return out


def torchyfy(sample, keys_to_props_map: dict[Props, str]) -> dict:
    torch_sample = {}
    for prop, key in keys_to_props_map.items():
        val = (
            sample[key].to(property_dtype(prop))
            if isinstance(sample[key], th.Tensor)
            else th.tensor(sample[key], dtype=property_dtype(prop))
        )
        if val.ndim == 0:  # scalars
            val = val.unsqueeze(0)
        torch_sample[prop] = val
    return torch_sample


def collate_fn(
    batch,
    props: list[Props],
    device=None,
    batch_func=batch_tall,
    pre_batch_preprocessors=None,
    post_batch_preprocessors=None,
) -> dict:
    if pre_batch_preprocessors is None:
        pre_batch_preprocessors = []
    if post_batch_preprocessors is None:
        post_batch_preprocessors = []
    if device is None:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
    batch = [torchyfy(sample, props) for sample in batch]
    n_atoms = th.tensor([len(sample[Props.atomic_numbers]) for sample in batch])
    props = list(props.keys())
    for func in pre_batch_preprocessors:
        batch, new_props = func(batch)
        props += new_props

    out = batch_func(batch, props, n_atoms)
    out[Props.n_atoms] = n_atoms
    out = {k: v.to(device, non_blocking=True) for k, v in out.items()}
    for func in post_batch_preprocessors:
        out = func(out)
    return out
