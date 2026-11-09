"""Cuadrículas de imágenes reales y generadas desde tensores del modelo."""
import matplotlib.pyplot as plt
import torch


def image_grid(images, columns=4, title="Muestras"):
    """Renderiza tensores [-1,1] sin sustituirlos por ejemplos estáticos."""
    if images is None or len(images) == 0:
        return None
    count = min(len(images), 16)
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.55, rows * 1.7))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for index, axis in enumerate(axes):
        axis.axis("off")
        if index < count:
            image = images[index].detach().cpu().squeeze()
            axis.imshow(image.clamp(-1, 1).add(1).div(2), cmap="gray", vmin=0, vmax=1)
    fig.suptitle(title); fig.tight_layout()
    return fig


def generate_images(generator, latent_dim, count, device, seed=None):
    """Genera nuevas imágenes del modelo usando ruido aleatorio opcionalmente sembrado."""
    generator.eval()
    generator_device = next(generator.parameters()).device
    generator = generator.to(device)
    random_generator = torch.Generator(device=device)
    if seed is not None:
        random_generator.manual_seed(seed)
    noise = torch.randn(count, latent_dim, device=device, generator=random_generator)
    with torch.no_grad():
        return generator(noise).cpu()
