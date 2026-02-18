import math
from typing import Literal

import torch as th
from md_et.nn.types import PropertyType, property_dims, property_type
from md_et.nn.types import Property as Props
from md_et.nn.dist import set_dtype
from torch import nn
from torch.autograd import grad
from torch.nn import functional as F

NODE_FEATURES_OFFSET = 128
MAX_ATOM_TYPE = 128
MAX_Z = 101


class PairEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        embd_dim: int,
        num_3d_kernels: int,
        cls_token: bool,
        num_heads: int,
        activation: str,
        ffn_multiplier: int,
        attention_dropout: float,
        ffn_dropout: float,
        head_dropout: float,
        norm_first: bool,
        norm: Literal["batch", "layer"],
        decomposer_type: Literal["pooling", "diagonal"],
        target_heads: list[str],
        head_project_down: bool,
        compose_dipole_from_charges: bool = False,
        use_electronic_embeddings: bool = True,
        energy_conserving: bool = False,
        directional_embed_type: Literal[
            "fourier", "direct", "spherical_harmonics"
        ] = "fourier",
        distance_embed_type: Literal["gaussian", "bessel"] = "gaussian",
        embed_edge_types: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = PairEmbedding(
            embd_dim,
            num_3d_kernels,
            distance_embed_type,
            embed_edge_types=embed_edge_types,
        )
        self.composer = Composer(embd_dim)
        self.decomposer = PoolingDecomposer(embd_dim)
        self.layers = nn.ModuleList(
            [
                EdgeTransformerLayer(
                    embd_dim,
                    num_heads,
                    ffn_dropout,
                    ffn_multiplier,
                    attention_dropout,
                )
                for _ in range(n_layers)
            ]
        )
        target_heads = [Props[t] for t in target_heads]

        self.heads = nn.ModuleDict(
            {
                str(target): NodeLevelRegressionHead(
                    target,
                    embd_dim=embd_dim,
                    head_dropout=head_dropout,
                    project_down=head_project_down,
                )
                for target in target_heads
            }
        )
        if energy_conserving:
            self.heads.pop(str(Props.forces))
            self.heads[str(Props.forces)] = EnergyConservingForcesHead(
                energy_head=self.heads[str(Props.formation_energy)],
            )
            self.heads.pop(str(Props.formation_energy))

    def reset_parameters(self) -> None:
        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs) -> dict:
        h, e, mask, positions = self.embedding(inputs)
        x = self.composer((h, e, mask))

        unbatch = mask.unsqueeze(2) * mask.unsqueeze(1)  # B x N x N
        x_mask = unbatch.unsqueeze(3) * mask.unsqueeze(1).unsqueeze(2)  # B x N x N x N

        for layer in self.layers:
            x = layer(x, x_mask)

        h = self.decomposer(x)
        out = {Props[k]: head(h, positions, mask) for k, head in self.heads.items()}
        out["embd"] = h
        return out


def cdist(x1, x2, eps=1e-8) -> th.Tensor:
    diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)  # (..., P, R, M)
    sq_dist = (diff**2).sum(dim=-1)  # (..., P, R)
    return th.sqrt(sq_dist + eps)


class PairEmbedding(nn.Module):
    def __init__(
        self,
        embd_dim: int,
        num_3d_kernels: int,
        distance_embed_type: Literal["gaussian", "bessel"] = "gaussian",
        embed_edge_types: bool = False,
    ) -> None:
        super().__init__()
        self.num_3d_kernels = num_3d_kernels
        self.embed_edge_types = embed_edge_types
        # node features
        self.nuclear_embedding = NuclearEmbedding(embd_dim)
        self.multiplicity_embed = nn.Embedding(NODE_FEATURES_OFFSET, embd_dim)
        self.charge_embed = nn.Embedding(NODE_FEATURES_OFFSET, embd_dim)
        nn.init.normal_(self.multiplicity_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.charge_embed.weight, mean=0.0, std=0.02)

        # pair features
        if distance_embed_type == "gaussian":
            self.m3d_embed = Gaussian3DEmbed(
                num_heads=embd_dim,
                num_edges=MAX_ATOM_TYPE**2
                if embed_edge_types
                else MAX_ATOM_TYPE * 2 + 1,  # backwards compatibility
                num_kernel=self.num_3d_kernels,
            )
        elif distance_embed_type == "bessel":
            self.m3d_embed = Bessel3DEmbed(
                n_rbf=self.num_3d_kernels,
                n_edge_types=MAX_ATOM_TYPE**2,
                out_dim=embd_dim,
            )
        else:
            raise ValueError(f"Invalid distance embed type {distance_embed_type}")

        self.directional_embed = DirectDirectionalEmbed(embd_dim)

    def forward(self, inputs) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        positions, atomic_numbers, mask, multiplicity, charge = (
            inputs[Props.positions],
            inputs[Props.atomic_numbers],
            inputs[Props.mask],
            inputs[Props.multiplicity],
            inputs[Props.charge],
        )

        h = self.nuclear_embedding(atomic_numbers.long())  # (b,n,e)

        multipl_embed = self.multiplicity_embed(multiplicity)  # (b,1,e)
        charge_embed = self.charge_embed(
            charge + (NODE_FEATURES_OFFSET // 2)
        )  # (b,1,e)
        g = multipl_embed + charge_embed  # (b,1,e)
        h += g

        # pair features
        D = cdist(positions, positions)  # (b,n,n)

        if self.embed_edge_types:
            n = atomic_numbers.size(1)
            atom_types = atomic_numbers.long().unsqueeze(-1)  # (b,n,1)
            atom_types_shifted = atom_types * 128  # (b,n,1)
            atom_types = atom_types.unsqueeze(2).expand(
                -1, -1, n, -1
            )  # (b,n,1,1) -> (b,n,n,1)
            atom_types_shifted = atom_types_shifted.unsqueeze(1).expand(
                -1, n, -1, -1
            )  # (b,1,n,1) -> (b,n,n,1)
            edge_types = atom_types + atom_types_shifted  # (b,n,n,1)
        else:
            edge_types = th.zeros_like(D).long().unsqueeze(-1)  # (b,n,n,1)

        e = self.m3d_embed(D, edge_types)  # (b,n,n,e)
        e += self.directional_embed(positions)  # (b,n,n,e)

        return h, e, mask, positions


class Composer(nn.Module):
    def __init__(
        self,
        embed_dim,
        linear: bool = True,
    ) -> None:
        super().__init__()
        concat_dim = 2 * embed_dim
        self.node_proj = MLP(concat_dim, embed_dim, linear=linear)

    def forward(self, inputs) -> th.Tensor:
        h, e, _ = inputs
        # create pair of node embeddings
        h_i = h.unsqueeze(2).expand(-1, -1, h.size(1), -1)  # (b,n,n,e)
        h_ij = th.cat([h_i, h_i.transpose(1, 2)], dim=-1)  # (b,n,n,2e)
        h_e = self.node_proj(h_ij)  # (b,n,n,e)

        x = e + h_e  # (b,n,n,e)
        return x


class PoolingDecomposer(nn.Module):
    def __init__(self, embd_dim) -> None:
        super().__init__()
        self.node_dim = embd_dim

        self.out_proj = MLP(embd_dim, 2 * embd_dim)
        self.node_mlp = MLP(embd_dim, embd_dim)

    def forward(self, x) -> th.Tensor:
        x1, x2 = self.out_proj(x).chunk(2, dim=-1)  # (b,n,n,2e)
        x = x1 + x2.transpose(1, 2)  # (b,n,n,e)
        x = x.sum(dim=2)  # (b,n,e)
        x = self.node_mlp(x)  # (b,n,e)
        return x


class FFN(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_multiplier: int = 2,
        dropout: float = 0,
    ) -> None:
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_multiplier * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.norm_aggregate = nn.LayerNorm(embed_dim)

        self.dropout_aggregate = nn.Dropout(dropout)
        self.embed_dim = embed_dim
        self.dropout = dropout

    # @th.compile
    def forward(self, x_prior, x) -> th.Tensor:
        x = self.dropout_aggregate(x)
        x = x_prior + x
        x = self.norm_aggregate(x)
        x = self.mlp(x) + x
        return self.norm(x)


class EdgeTransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        ffn_multiplier: int,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.attention = FastEdgeAttention(embed_dim, num_heads, attention_dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, ffn_multiplier, dropout)

    def forward(self, x_in, mask=None) -> th.Tensor:
        x = self.norm(x_in)

        x_upd = (
            self.attention(x, x, x, ~mask)
            if mask is not None
            else self.attention(x, x, x)
        )
        x = self.ffn(x_in, x_upd)
        return x


def triang_attn(q, k) -> th.Tensor:
    out = q.unsqueeze(3) * k.unsqueeze(1)
    return out.sum(dim=5)


def val_fusion(v1, v2) -> th.Tensor:
    return v1.unsqueeze(3) * v2.unsqueeze(1)


def final_comp(att, val) -> th.Tensor:
    out = att.unsqueeze(-1) * val
    return out.sum(dim=2)


class FastEdgeAttention(nn.Module):
    def __init__(
        self, embed_dim, num_heads, dropout, dtype: th.dtype | None = None
    ) -> None:
        super().__init__()
        self.dtype = set_dtype(dtype)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads

        self.qlin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.klin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v1lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v2lin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.olin = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    # @th.compile
    def forward(self, query, key, value, mask=None) -> th.Tensor:
        num_batches = query.size(0)
        num_nodes_q = query.size(1)
        num_nodes_k = key.size(1)

        left_k = self.qlin(query)
        right_k = self.klin(key)
        left_v = self.v1lin(value)
        right_v = self.v2lin(value)

        left_k = left_k.view(
            num_batches, num_nodes_q, num_nodes_q, self.num_heads, self.d_k
        ) / math.sqrt(self.d_k)
        right_k = right_k.view(
            num_batches, key.size(1), key.size(2), self.num_heads, self.d_k
        )
        left_v = left_v.view_as(right_k)
        right_v = right_v.view_as(right_k)

        if hasattr(self, "norms"):
            left_k = self.norms[0](left_k)
            right_k = self.norms[1](right_k)

        scores = triang_attn(left_k, right_k)

        if mask is not None:
            scores_dtype = scores.dtype
            scores = (
                scores.to(self.dtype)
                .masked_fill(mask.unsqueeze(4), -1e9)
                .to(scores_dtype)
            )

        scores = scores - scores.max(dim=2, keepdim=True).values
        att = F.softmax(scores, dim=2)
        att = self.dropout(att)
        val = val_fusion(left_v, right_v)

        if hasattr(self, "norms"):
            val = self.norms[2](val)

        x = final_comp(att, val)
        x = x.view(num_batches, num_nodes_q, num_nodes_k, self.embed_dim)
        return self.olin(x)


class MLP(nn.Sequential):
    def __init__(
        self, input_dim, output_dim, dropout: float = 0.0, linear: bool = False
    ) -> None:
        if not linear:
            hidden_dim = output_dim

            layers = [
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Dropout(dropout),
            ]
            super().__init__(*layers)
        else:
            super().__init__(
                nn.Linear(input_dim, output_dim),
            )


def get_output_mlp(
    embd_dim: int,
    project_down: bool,
    head_dropout: float,
    output_dim: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(embd_dim, embd_dim // 2 if project_down else embd_dim),
        nn.GELU(),
        nn.Dropout(head_dropout),
        nn.Linear(
            embd_dim // 2 if project_down else embd_dim,
            embd_dim // 4 if project_down else embd_dim,
        ),
        nn.GELU(),
        nn.Dropout(head_dropout),
        nn.Linear(embd_dim // 4 if project_down else embd_dim, output_dim, bias=False),
    )


class NodeLevelRegressionHead(nn.Module):
    def __init__(
        self,
        target: Props,
        embd_dim: int,
        head_dropout: float,
        project_down: bool,
        dtype: th.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dtype = set_dtype(dtype)
        self.embd_dim = embd_dim
        self.project_down = project_down
        self.head_dropout = head_dropout
        self.final_ln_node = nn.LayerNorm(embd_dim)
        self.target = target
        self.output_dim = property_dims[target]
        self.mlp = get_output_mlp(
            embd_dim, project_down, head_dropout, self.output_dim
        )
        self.target_type: PropertyType = property_type[target]
        assert self.target_type in [
            PropertyType.mol_wise,
            PropertyType.atom_wise,
        ], f"Invalid target type {self.target_type}"

    def reset_parameters(self) -> None:
        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, h, positions, mask) -> th.Tensor:
        h = h.clone()  # (b,n,e)
        h = self.final_ln_node(h)  # (b,n,e)
        mask = mask.to(self.dtype).unsqueeze(-1)

        if self.target_type == PropertyType.mol_wise:
            h = (h * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)  # (b,e)
        elif self.target_type == PropertyType.atom_wise:
            h = h * mask

        return self.mlp(h)


class EnergyConservingForcesHead(nn.Module):
    def __init__(
        self, energy_head: NodeLevelRegressionHead, return_energy: bool = False
    ) -> None:
        super().__init__()
        self.energy_head = energy_head
        self.return_energy = return_energy

    def forward(self, h, positions, mask) -> th.Tensor:
        energies = self.energy_head(h, positions, mask).squeeze()
        dEdpos = th.zeros_like(positions)
        energy_sum = energies.sum()
        dEdpos = grad(energy_sum, positions, create_graph=True)[0] * mask.unsqueeze(-1)
        if self.return_energy:
            return dEdpos, energies.detach()
        if not self.training:
            dEdpos = dEdpos.detach()
        return dEdpos


class NuclearEmbedding(nn.Module):
    def __init__(self, embedding_dim, zero_init=True) -> None:
        super().__init__()
        self.embedding = nn.Embedding(MAX_Z + 1, embedding_dim)
        self.reset_parameters(zero_init)

    def reset_parameters(self, zero_init=True) -> None:
        if zero_init:
            nn.init.zeros_(self.embedding.weight)
        else:
            nn.init.uniform_(self.embedding, -math.sqrt(3), math.sqrt(3))

    def forward(self, atomic_numbers) -> th.Tensor:
        return self.embedding(atomic_numbers)


class DirectDirectionalEmbed(nn.Module):
    def __init__(self, embedding_dim, eps=1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.linear = nn.Linear(3, 2 * embedding_dim)
        self.linear2 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.linear.weight.data.normal_(0, 0.001)
        self.linear2.weight.data.normal_(0, 0.001)

    def forward(self, positions) -> th.Tensor:  # b, n, 3
        _, n, _ = positions.shape  # positions: (b, n, 3)
        positions_i = positions.unsqueeze(2).expand(-1, -1, n, -1)  # (b, n, n, 3)
        positions_j = positions.unsqueeze(1).expand(-1, n, -1, -1)  # (b, n, n, 3)
        directions = positions_j - positions_i  # (b, n, n, 3)
        distances = th.norm(directions, dim=-1, keepdim=True)  # (b, n, n, 1)
        normed_directions = directions / (distances + self.eps)  # (b, n, n, 3)
        x = self.linear(normed_directions)  # (b, n, n, 2 * embedding_dim)
        x = F.gelu(x)
        x = self.linear2(x)
        x = F.gelu(x)
        return x


class NonLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden=None) -> None:
        super().__init__()
        if hidden is None:
            hidden = input_size
        self.layer1 = nn.Linear(input_size, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x) -> th.Tensor:
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x


class Gaussian3DEmbed(nn.Module):
    def __init__(self, num_heads: int, num_edges: int, num_kernel: int) -> None:
        super().__init__()
        self.gbf = GaussianLayer(num_kernel, num_edges)
        self.gbf_proj = NonLinear(num_kernel, num_heads)

    def forward(self, dist, node_type_edge) -> th.Tensor:
        edge_feature = self.gbf(dist, node_type_edge.long())  # (b, n, n, K)
        gbf_result = self.gbf_proj(edge_feature)  # (b, n, n, H)
        return gbf_result


class Bessel3DEmbed(nn.Module):
    def __init__(
        self, n_rbf: int, n_edge_types: int, out_dim: int, cutoff: float = 25
    ) -> None:
        super().__init__()
        self.n_rbf = n_rbf
        self.freqs = nn.Embedding(n_edge_types, n_rbf)
        nn.init.uniform_(self.freqs.weight, 1.0, math.pi * n_rbf / cutoff)
        self.linear = nn.Linear(n_rbf, out_dim)

    def forward(
        self,
        dist: th.Tensor,  # (b, n, n)
        edge_types: th.Tensor,  # (b, n, n, 1)
    ) -> th.Tensor:  # (b, n, n, n_rbf)
        dist = dist.unsqueeze(-1)
        ax = dist * self.freqs(edge_types).squeeze()  # (b, n, n, n_rbf)
        sinax = th.sin(ax)  # (b, n, n, n_rbf)
        norm = dist.clamp(min=1e-5)  # (b, n, n, 1)
        y = sinax / norm  # (b, n, n, n_rbf)
        y = self.linear(y)  # (b, n, n, out_dim)
        return y  # (b, n, n, out_dim)


@th.jit.script
def gaussian(x, mean, std) -> th.Tensor:
    pi = math.pi
    a = (2 * pi) ** 0.5
    return th.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class GaussianLayer(nn.Module):
    def __init__(
        self, K=128, edge_types=512 * 3, dtype: th.dtype | None = None
    ) -> None:
        super().__init__()
        self.K = K
        self.dtype = set_dtype(dtype)
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 7)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types) -> th.Tensor:
        mul = self.mul(edge_types).sum(dim=-2)
        bias = self.bias(edge_types).sum(dim=-2)
        x = mul * x.unsqueeze(-1) + bias
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.to(self.dtype).view(-1)
        std = self.stds.weight.to(self.dtype).view(-1).abs() + 1e-2
        return gaussian(x.to(self.dtype), mean, std).type_as(self.means.weight)
