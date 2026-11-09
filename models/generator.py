"""Generador MLP de imágenes MNIST."""
import torch
from torch import nn


class Generator(nn.Module):
    """Convierte vectores latentes en imágenes de dígitos de 28×28."""

    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        # El vector latente resume 100 variables aleatorias; sus combinaciones
        # ofrecen al modelo grados de libertad para representar estilos variados.
        # Las capas densas amplían esa representación hasta los 784 píxeles.
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256),  # Proyecta el ruido a rasgos iniciales.
            nn.LeakyReLU(0.2, inplace=True),  # Conserva gradiente para entradas negativas.
            nn.Linear(256, 512),  # Combina rasgos en patrones más complejos.
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),  # Construye una representación rica del dígito.
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),  # Produce un valor para cada píxel.
            nn.Tanh(),  # Acota píxeles a [-1, 1], igual que el MNIST normalizado.
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Transforma ruido [lote, latent_dim] en imágenes [lote, 1, 28, 28]."""
        flat = self.network(noise)
        return flat.view(noise.shape[0], 1, 28, 28)

    def sample_noise(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Muestrea nuevos vectores z de una distribución normal estándar."""
        return torch.randn(batch_size, self.latent_dim, device=device)


# Instanciación centralizada en training.trainer.create_models.
