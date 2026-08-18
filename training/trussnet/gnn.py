"""A graph network that learns the truss solve, shaped like the solver itself.

Why message passing rather than convolution
-------------------------------------------
The answer is u = K^-1 f. K is sparse -- one block per member -- but its inverse
is dense: every node's displacement depends on every other node. A feedforward
local operator is the wrong computational class for that, which is why widening
the CNN's receptive field to cover the support span made it *worse* rather than
better (measured 2026-08-18: 405px field scored 0.67 points below a 237px one).

Iterative solvers for K u = f -- Jacobi, Gauss-Seidel, conjugate gradient -- are
message passing on K's sparsity graph. So the processor here is that, learned:
T rounds of exchange along the members, with weights *shared* across rounds so
a round is genuinely an iteration rather than a layer. Two consequences follow.
Running fewer rounds at inference than in training yields a less-converged, and
therefore weaker but still physically coherent, opponent -- difficulty as solver
iterations. And the graph diameter is 3, so information crosses any truss in 3
rounds; more rounds buy accuracy, not reach.

The edge features are the exact entries of the element stiffness matrix,
(1/L_hat) * [[c^2, cs], [cs, s^2]]. The network is handed the ingredients of K
and has only to learn to invert it, never to rediscover it from geometry.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .graph_data import EDGE_FEATURES, NODE_FEATURES


def mlp(sizes, out_activation=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or out_activation:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class TrussGNN(nn.Module):
    def __init__(self, hidden=64, rounds=10, shared=True):
        super().__init__()
        self.rounds, self.shared = rounds, shared

        self.node_encoder = mlp([NODE_FEATURES, hidden, hidden], out_activation=True)
        self.edge_encoder = mlp([EDGE_FEATURES, hidden, hidden], out_activation=True)

        blocks = 1 if shared else rounds
        self.message = nn.ModuleList(
            mlp([3 * hidden, hidden, hidden], out_activation=True) for _ in range(blocks)
        )
        self.update = nn.ModuleList(
            mlp([2 * hidden, hidden, hidden], out_activation=True) for _ in range(blocks)
        )
        # LayerNorm keeps an unrolled, weight-shared recurrence from drifting in
        # scale over its iterations -- the failure mode a deep shared stack has
        # that a plain feedforward net does not.
        self.norm = nn.LayerNorm(hidden)
        self.decoder = mlp([hidden, hidden, 2])

    def forward(self, batch, rounds=None):
        rounds = self.rounds if rounds is None else rounds
        node_feat = batch['node_feat']
        edge_index, edge_feat = batch['edge_index'], batch['edge_feat']
        edge_mask = batch['edge_mask'].unsqueeze(-1)

        h = self.node_encoder(node_feat)
        e = self.edge_encoder(edge_feat) * edge_mask

        src = edge_index[..., 0].unsqueeze(-1).expand(-1, -1, h.shape[-1])
        dst = edge_index[..., 1].unsqueeze(-1).expand(-1, -1, h.shape[-1])

        for t in range(rounds):
            block = 0 if self.shared else t
            m = self.message[block](torch.cat([h.gather(1, src), h.gather(1, dst), e], dim=-1))
            # Sum, not mean: assembling K sums the contribution of every member
            # meeting at a node, so a node with more members really is stiffer.
            agg = torch.zeros_like(h).scatter_add_(1, dst, m * edge_mask)
            h = self.norm(h + self.update[block](torch.cat([h, agg], dim=-1)))

        # Supports are fully pinned. That is a boundary condition, not something
        # to be learned approximately, so it is imposed exactly.
        return self.decoder(h) * batch['free_mask'].unsqueeze(-1)


def physics_residual(u_hat, batch):
    """How far a predicted displacement field is from equilibrium.

    Needs no labels: it is the non-dimensionalised K u - f evaluated on the
    graph the model was given. Returns the per-node residual for free nodes,
    scaled by the applied load so it reads as a fraction of that load.
    """
    edge_index, edge_feat = batch['edge_index'], batch['edge_feat']
    edge_mask = batch['edge_mask'].unsqueeze(-1)
    src, dst = edge_index[..., 0], edge_index[..., 1]

    idx_src = src.unsqueeze(-1).expand(-1, -1, 2)
    idx_dst = dst.unsqueeze(-1).expand(-1, -1, 2)
    rel = u_hat.gather(1, idx_dst) - u_hat.gather(1, idx_src)

    cc, cs, ss, inv_len = (edge_feat[..., i] for i in range(4))
    # (1 / L_hat) * [[c^2, cs], [cs, s^2]] @ (u_j - u_i), the member's pull on
    # the node it points at.
    fx = inv_len * (cc * rel[..., 0] + cs * rel[..., 1])
    fy = inv_len * (cs * rel[..., 0] + ss * rel[..., 1])
    internal = torch.stack([fx, fy], dim=-1) * edge_mask

    # `internal` accumulated at dst is (K u)_i; equilibrium wants it to equal
    # the applied load, so the residual subtracts. Verified against the exact
    # field: this form gives ~1e-6, the opposite sign gives exactly 2f.
    net = torch.zeros_like(u_hat).scatter_add_(1, idx_dst, internal)
    external = batch['node_feat'][..., 3:5]
    residual = (net - external) * batch['free_mask'].unsqueeze(-1)
    return residual / batch['f_hat'].view(-1, 1, 1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
