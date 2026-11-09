"""Ciclo de entrenamiento independiente de Streamlit."""
import copy
import time
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from models import Generator, Discriminator
from training.losses import discriminator_loss, generator_loss
from training.metrics import METRIC_COLUMNS, make_epoch_record


def create_models(latent_dim: int, device: torch.device):
    """Instancia explícitamente G y D y los mueve al dispositivo elegido."""
    generator = Generator(latent_dim).to(device)  # Generador: ruido -> imagen.
    discriminator = Discriminator().to(device)  # Discriminador: imagen -> logit real/falso.
    return generator, discriminator


def save_checkpoint(path: Path, generator, discriminator, g_optimizer, d_optimizer,
                    config: dict, epoch: int, history: list, fixed_noise: torch.Tensor):
    """Guarda el estado completo para inspección o reanudación posterior."""
    payload = {"format_version": 1, "generator": generator.state_dict(),
               "discriminator": discriminator.state_dict(), "g_optimizer": g_optimizer.state_dict(),
               "d_optimizer": d_optimizer.state_dict(), "config": config, "epoch": epoch,
               "history": history, "fixed_noise": fixed_noise.cpu(),
               "torch_rng_state": torch.get_rng_state(),
               "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, generator, discriminator, g_optimizer=None, d_optimizer=None,
                    map_location="cpu"):
    """Restaura los tensores y estados guardados; devuelve configuración e historial."""
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # Compatibilidad con versiones de PyTorch anteriores a weights_only.
        payload = torch.load(path, map_location=map_location)
    if payload.get("format_version") != 1:
        raise ValueError("Versión de checkpoint no compatible.")
    generator.load_state_dict(payload["generator"])
    discriminator.load_state_dict(payload["discriminator"])
    if g_optimizer is not None:
        g_optimizer.load_state_dict(payload["g_optimizer"])
    if d_optimizer is not None:
        d_optimizer.load_state_dict(payload["d_optimizer"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    return payload


def train(config, loader, run_dir: Path, device: torch.device, progress_callback=None,
          stop_event=None, initial_state=None):
    """Entrena épocas y emite snapshots seguros para la interfaz interactiva."""
    config.validate()
    run_dir.mkdir(parents=True, exist_ok=True)
    generator, discriminator = create_models(config.latent_dim, device)
    # Inicialización normal estándar de las matrices densas; biases en cero.
    for model in (generator, discriminator):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.02)
                nn.init.zeros_(module.bias)
    g_optimizer = Adam(generator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    d_optimizer = Adam(discriminator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    start_epoch, history = 0, []
    fixed_noise = torch.randn(config.sample_count, config.latent_dim, device=device)
    if initial_state:
        payload = load_checkpoint(initial_state, generator, discriminator, g_optimizer, d_optimizer, device)
        start_epoch, history, fixed_noise = payload["epoch"], payload["history"], payload["fixed_noise"].to(device)
    initial_weights = {"generator": copy.deepcopy(generator.state_dict()),
                       "discriminator": copy.deepcopy(discriminator.state_dict())}
    torch.save(initial_weights, run_dir / "initial_weights.pt")
    started = time.perf_counter()
    images_processed = history[-1]["images_processed"] if history else 0
    for epoch in range(start_epoch, config.epochs):
        generator.train(); discriminator.train()
        totals = {"g": 0.0, "d": 0.0, "real": 0.0, "fake": 0.0, "n": 0}
        for real_images, _labels in loader:
            if stop_event is not None and stop_event.is_set():
                break
            real_images = real_images.to(device)
            batch = real_images.shape[0]
            # Actualización D: imágenes falsas separadas del grafo de G por detach().
            d_optimizer.zero_grad(set_to_none=True)
            real_logits = discriminator(real_images)
            with torch.no_grad():
                fake_images = generator(generator.sample_noise(batch, device))
            fake_logits = discriminator(fake_images.detach())
            d_loss = discriminator_loss(real_logits, fake_logits)
            d_loss.backward()
            d_optimizer.step()
            # Actualización G: los pesos de D se congelan, pero el gradiente
            # atraviesa D hasta G para indicar cómo mejorar la imagen generada.
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            g_optimizer.zero_grad(set_to_none=True)
            generated = generator(generator.sample_noise(batch, device))
            g_logits = discriminator(generated)
            g_loss = generator_loss(g_logits)
            g_loss.backward()
            g_optimizer.step()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)
            totals["g"] += g_loss.item() * batch; totals["d"] += d_loss.item() * batch
            totals["real"] += torch.sigmoid(real_logits).mean().item() * batch
            totals["fake"] += torch.sigmoid(fake_logits).mean().item() * batch
            totals["n"] += batch; images_processed += batch
        if totals["n"] == 0:
            break
        if stop_event is not None and stop_event.is_set():
            break
        record = make_epoch_record(epoch + 1, totals["g"] / totals["n"], totals["d"] / totals["n"],
                                   time.perf_counter() - started, images_processed,
                                   totals["real"] / totals["n"], totals["fake"] / totals["n"])
        history.append(record)
        pd.DataFrame(history, columns=METRIC_COLUMNS).to_csv(run_dir / "metrics.csv", index=False)
        generator.eval()
        with torch.no_grad():
            samples = generator(fixed_noise).detach().cpu()
        torch.save(samples, run_dir / f"samples_epoch_{epoch + 1:03d}.pt")
        save_checkpoint(run_dir / "checkpoint.pt", generator, discriminator, g_optimizer,
                        d_optimizer, config.to_dict(), epoch + 1, history, fixed_noise)
        if progress_callback:
            progress_callback({"status": "running", "epoch": epoch + 1, "total_epochs": config.epochs,
                               "metrics": record, "history": list(history),
                               "generator": copy.deepcopy(generator.state_dict()),
                               "discriminator": copy.deepcopy(discriminator.state_dict()),
                               "samples": samples})
        if stop_event is not None and stop_event.is_set():
            break
    status = "detenido" if stop_event is not None and stop_event.is_set() else "completado"
    return {"status": status, "history": history, "run_dir": str(run_dir),
            "generator": generator, "discriminator": discriminator}
