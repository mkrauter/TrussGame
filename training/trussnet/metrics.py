"""The game's scoring rule, and the baselines any model has to beat.

The metric is relative: the error is divided by how far the node actually
travelled. So it measures the fraction of the movement a prediction captured,
not a distance in pixels -- which is why guessing the starting position scores
exactly zero however small the movement was.
"""
from __future__ import annotations

import numpy as np


def accuracy(start, end, guess):
    """Score in 0..100, vectorised over a leading sample axis.

    100 means landing exactly on the settled position; 0 means missing by at
    least as far as the node travelled.
    """
    start, end, guess = (np.asarray(a, dtype=np.float64) for a in (start, end, guess))
    travelled = np.linalg.norm(end - start, axis=-1)
    missed = np.linalg.norm(guess - end, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        score = (1.0 - missed / travelled) * 100.0
    # A node that never moved cannot be predicted; the game scores it zero.
    return np.where(travelled == 0, 0.0, np.maximum(score, 0.0))


def summarise(scores):
    scores = np.asarray(scores, dtype=np.float64)
    return {
        'mean': float(scores.mean()),
        'median': float(np.median(scores)),
        'std': float(scores.std()),
        'p25': float(np.percentile(scores, 25)),
        'p75': float(np.percentile(scores, 75)),
        'zeros': float((scores == 0).mean() * 100),
    }


def support_span(support_a, support_b):
    return np.linalg.norm(np.asarray(support_b) - np.asarray(support_a), axis=-1)


def fit_baselines(train):
    """Constants fitted on the training split only, so evaluation stays honest."""
    delta = train['end'] - train['start']
    travel = np.linalg.norm(delta, axis=-1)
    span = support_span(train['support_a'], train['support_b'])
    return {
        'mean_delta': delta.mean(axis=0),
        'mean_travel': float(travel.mean()),
        # Displacement scales exactly with the structure's size (K is
        # proportional to EA/L), so travel expressed in support spans should be
        # a tighter constant than travel in pixels.
        'mean_travel_over_span': float((travel / span).mean()),
    }


def baseline_predictions(split, fitted):
    """Every prediction is start + a displacement guess.

    All of these are given the true starting position. That is deliberate: the
    deployed model localises the node at 0.97 correlation, so localisation is
    effectively solved and the interesting question is purely the displacement.
    """
    start = split['start']
    span = support_span(split['support_a'], split['support_b'])
    down = np.array([0.0, 1.0])

    preds = {
        'click the starting point': start,
        'constant mean displacement': start + fitted['mean_delta'],
        'straight down, mean travel': start + down * fitted['mean_travel'],
        'straight down, travel scaled by span':
            start + down * (span * fitted['mean_travel_over_span'])[:, None],
    }

    # Diagnostics, not fair baselines -- these are told part of the answer.
    travel = np.linalg.norm(split['end'] - split['start'], axis=-1)
    preds['[oracle] straight down, true magnitude'] = start + down * travel[:, None]
    preds['[oracle] exact'] = split['end']
    return preds
