"""Pérdidas BCE estables para el juego adversario."""
import torch
from torch import nn


_bce = nn.BCEWithLogitsLoss()


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """D suma BCE(real,1)+BCE(fake,0); BCEWithLogits incorpora sigmoid estable."""
    real_targets = torch.ones_like(real_logits)
    fake_targets = torch.zeros_like(fake_logits)
    return _bce(real_logits, real_targets) + _bce(fake_logits, fake_targets)


def generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """G etiqueta como reales sus falsos para aprender a engañar a D."""
    return _bce(fake_logits, torch.ones_like(fake_logits))
