"""Selección automática del acelerador."""
import torch


def get_device() -> torch.device:
    """Prioriza CUDA cuando está disponible; en caso contrario usa CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
