"""Gráficas didácticas para pérdidas y métricas de la GAN."""
import matplotlib.pyplot as plt
import pandas as pd


def loss_figure(history):
    """Dibuja cada pérdida en paneles separados para mantener una sola escala por eje."""
    if not history:
        return None
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(frame["epoch"], frame["generator_loss"], color="#2864dc", linewidth=2)
    axes[0].set(title="Pérdida del generador", xlabel="Época", ylabel="BCE")
    axes[1].plot(frame["epoch"], frame["discriminator_loss"], color="#bc5a24", linewidth=2)
    axes[1].set(title="Pérdida del discriminador", xlabel="Época", ylabel="BCE")
    for axis in axes:
        axis.grid(alpha=0.18)
    fig.tight_layout()
    return fig


def probability_figure(history):
    """Grafica probabilidades promedio reales y falsas para seguir el equilibrio."""
    if not history:
        return None
    frame = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(frame["epoch"], frame["real_probability"], label="Imagen real", color="#18856b")
    ax.plot(frame["epoch"], frame["fake_probability"], label="Imagen generada", color="#a34687")
    ax.set(xlabel="Época", ylabel="Probabilidad media", ylim=(0, 1), title="Predicciones de D")
    ax.grid(alpha=0.18); ax.legend(); fig.tight_layout()
    return fig
