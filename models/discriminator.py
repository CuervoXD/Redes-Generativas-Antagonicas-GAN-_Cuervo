"""Discriminador MLP que devuelve logits para imágenes MNIST."""
import torch
from torch import nn


class Discriminator(nn.Module):
    """Evalúa si cada imagen parece real; no aplica sigmoid internamente."""

    def __init__(self):
        super().__init__()
        # Aplana 28×28 píxeles y aprende progresivamente rasgos discriminativos.
        self.network = nn.Sequential(
            nn.Linear(784, 512),  # Primer nivel de combinación de píxeles.
            nn.LeakyReLU(0.2, inplace=True),  # Evita gradientes nulos en zona negativa.
            nn.Linear(512, 256),  # Resume la evidencia visual relevante.
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),  # Logit: alto favorece real, bajo favorece falso.
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Acepta [lote, 1, 28, 28] o [lote, 784] y devuelve logits [lote, 1]."""
        flat = images.view(images.shape[0], 784)
        return self.network(flat)

    @staticmethod
    def probability(logits: torch.Tensor) -> torch.Tensor:
        """Convierte logits en probabilidades solo para visualización/métricas."""
        return torch.sigmoid(logits)


# Instanciación centralizada en training.trainer.create_models.
