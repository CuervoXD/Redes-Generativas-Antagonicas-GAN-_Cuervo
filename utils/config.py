"""Configuración validada y semilla de las ejecuciones."""
from dataclasses import asdict, dataclass
import random
import numpy as np
import torch


@dataclass
class TrainConfig:
    category: int = 7
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 0.0002
    latent_dim: int = 100
    seed: int = 42
    sample_count: int = 16
    num_workers: int = 0

    def validate(self) -> None:
        if not 0 <= self.category <= 9:
            raise ValueError("La categoría debe estar entre 0 y 9.")
        if self.epochs < 1 or self.batch_size < 1 or self.latent_dim < 1:
            raise ValueError("Épocas, batch size y dimensión latente deben ser positivos.")
        if self.learning_rate <= 0:
            raise ValueError("El learning rate debe ser positivo.")

    def to_dict(self):
        return asdict(self)


def seed_everything(seed: int) -> None:
    """Fija los generadores aleatorios disponibles para facilitar reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
