"""Carga, normalización y filtrado de MNIST por categoría."""
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_mnist_loader(category: int, batch_size: int, root: str = "data/mnist", num_workers: int = 0):
    """Descarga MNIST automáticamente y crea un loader solo con la etiqueta elegida."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [0,1] -> [-1,1], compatible con Tanh.
    ])
    dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    indices = (dataset.targets == category).nonzero(as_tuple=True)[0].tolist()
    if not indices:
        raise ValueError(f"No se encontraron ejemplos MNIST para la clase {category}.")
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=num_workers, pin_memory=False), subset
