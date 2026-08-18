"""Loading the structured corpus produced by generate_graph_corpus.mjs.

Every truss in this game has exactly 10 nodes and 18-23 members, so the graph
can be batched as dense padded tensors and no graph library is needed. Edges
are stored in both directions because message passing sends along each member
twice, once toward each end.

Non-dimensionalisation
----------------------
A linear truss is scale-equivariant: multiply every coordinate by alpha and the
stiffness k = EA/L scales as 1/alpha, so displacements scale as alpha. Dividing
lengths by the support span therefore removes an entire degree of freedom from
what the network has to learn, and the target becomes travel-per-span -- the
same quantity the span-scaled baseline used.

With u = span * u_hat and L = span * L_hat, equilibrium

    sum_j (EA / L) M (u_i - u_j) = f_i,      M = [[c^2, cs], [cs, s^2]]

reduces to

    sum_j (1 / L_hat) M (u_hat_i - u_hat_j) = f_i / EA

so the only physical constant that survives is f_hat = force / (E * A). That is
what `physics_residual` in gnn.py checks, and it needs no labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / 'graph_corpus'

# Node feature layout, kept explicit so the model and any debugging agree.
NODE_FEATURES = 7   # is_free, is_support, is_loaded, fx, fy, x, y
EDGE_FEATURES = 4   # c^2, cs, s^2, 1 / L_hat


def load_raw(split, root=DEFAULT_ROOT):
    payload = json.loads((Path(root) / split / 'graphs.json').read_text(encoding='utf-8'))
    return payload['meta'], payload['samples']


class TrussGraphs(Dataset):
    """Trusses as graphs, with perception modelled as coordinate jitter.

    `sigma` is the standard deviation, in screen pixels, of the error with which
    the model is assumed to read node positions off the canvas. It is the
    fairness dial: 0 gives the network exact geometry, ~2-3 gives it roughly
    what a human eye extracts from a 900px canvas. Measured cost of exact over
    human-level perception is about 2 points of game score, so this is a small
    correction -- but an explicit one, unlike the resolution of a screenshot.
    """

    def __init__(self, split, root=DEFAULT_ROOT, sigma=0.0, seed=0, jitter=True,
                 member_noise=0.0):
        meta, samples = load_raw(split, root)
        self.meta = meta
        self.sigma = float(sigma)
        self.jitter = jitter
        # Probability that a sample's member list is perturbed by one edge --
        # dropped or invented. The detector reads connectivity off the frame and
        # gets it exactly right on ~88% of frames, so a model trained only on
        # perfect graphs is being trained on inputs it will not receive.
        self.member_noise = float(member_noise)
        self._rng = np.random.default_rng(seed)

        self.f_hat = meta['physics']['force'] / (meta['physics']['E'] * meta['physics']['A'])

        self.nodes = np.array([s['nodes'] for s in samples], dtype=np.float64)
        self.displacement = np.array([s['displacement'] for s in samples], dtype=np.float64)
        self.loaded = np.array([s['loadedNode'] for s in samples], dtype=np.int64)
        self.supports = np.array([s['supports'] for s in samples], dtype=np.int64)

        # Members are undirected; store both directions, padded to the corpus
        # maximum so every sample is the same shape and collates for free.
        self.elements = [[tuple(e) for e in s['elements']] for s in samples]
        # One slot spare, so an invented member has somewhere to go.
        self.max_edges = 2 * (max(len(e) for e in self.elements) + 1)

    def __len__(self):
        return len(self.nodes)

    def _perceived_elements(self, i, node_count):
        """The member list as the model receives it, occasionally one edge off.

        The label stays the true displacement, so the model is being asked to
        give the right answer from a slightly wrong structure -- which is the
        situation the detector actually puts it in, and which it cannot be
        robust to if it only ever trains on perfect graphs.
        """
        elements = self.elements[i]
        if self.member_noise <= 0 or self._rng.random() >= self.member_noise:
            return list(elements)

        elements = list(elements)
        if self._rng.random() < 0.5 and len(elements) > 1:
            elements.pop(int(self._rng.integers(len(elements))))
        else:
            present = {tuple(sorted(e)) for e in elements}
            for _ in range(8):
                a, b = self._rng.integers(0, node_count, 2)
                if a != b and tuple(sorted((int(a), int(b)))) not in present:
                    elements.append((int(a), int(b)))
                    break
        return elements

    def __getitem__(self, i):
        nodes = self.nodes[i]
        support = self.supports[i]
        loaded = int(self.loaded[i])

        # What the model sees. Connectivity, supports and the loaded node are
        # drawn unambiguously on screen, so only the coordinates are uncertain.
        seen = nodes
        if self.jitter and self.sigma > 0:
            seen = nodes + self._rng.normal(0.0, self.sigma, nodes.shape)

        span = float(np.linalg.norm(seen[support[1]] - seen[support[0]]))
        centre = (seen[support[0]] + seen[support[1]]) / 2.0
        pos = (seen - centre) / span

        is_support = np.zeros(len(nodes), dtype=np.float64)
        is_support[support] = 1.0
        is_loaded = np.zeros(len(nodes), dtype=np.float64)
        is_loaded[loaded] = 1.0
        force = np.zeros_like(nodes)
        force[loaded, 1] = self.f_hat          # load is +y, downward on screen

        node_feat = np.concatenate(
            [(1.0 - is_support)[:, None], is_support[:, None], is_loaded[:, None], force, pos],
            axis=1,
        )

        elements = self._perceived_elements(i, len(nodes))
        edge_index = np.zeros((self.max_edges, 2), dtype=np.int64)
        edge_mask = np.zeros(self.max_edges, dtype=np.float32)
        pairs = elements + [(b, a) for a, b in elements]
        edge_index[:len(pairs)] = pairs
        edge_mask[:len(pairs)] = 1.0

        src, dst = edge_index[:, 0], edge_index[:, 1]
        delta = pos[dst] - pos[src]
        length = np.linalg.norm(delta, axis=1)
        safe = np.where(length > 0, length, 1.0)
        c, s = delta[:, 0] / safe, delta[:, 1] / safe
        edge_feat = np.stack([c * c, c * s, s * s, 1.0 / safe], axis=1)
        edge_feat *= edge_mask[:, None]

        # Targets are the true displacements, non-dimensionalised by the span
        # the model believes it sees -- the same quantity it predicts.
        target = self.displacement[i] / span

        return {
            'node_feat': torch.from_numpy(node_feat).float(),
            'edge_index': torch.from_numpy(edge_index),
            'edge_feat': torch.from_numpy(edge_feat).float(),
            'edge_mask': torch.from_numpy(edge_mask),
            'free_mask': torch.from_numpy(1.0 - is_support).float(),
            'target': torch.from_numpy(target).float(),
            'f_hat': torch.tensor(self.f_hat, dtype=torch.float32),
            'loaded': torch.tensor(loaded),
            'span': torch.tensor(span, dtype=torch.float32),
            # Screen-pixel quantities, for scoring on the game's own metric.
            'seen_start': torch.from_numpy(seen[loaded]).float(),
            'true_start': torch.from_numpy(nodes[loaded]).float(),
            'true_end': torch.from_numpy(nodes[loaded] + self.displacement[i][loaded]).float(),
        }
