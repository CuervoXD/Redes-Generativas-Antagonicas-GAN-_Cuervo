"""Pruebas unitarias rápidas, sin descargar MNIST."""
import torch
from models import Generator, Discriminator
from training.losses import discriminator_loss, generator_loss
from training.trainer import create_models, train
from utils.config import TrainConfig
from visualization.parameters import parameter_count
from torch.utils.data import DataLoader, TensorDataset


def test_model_shapes_and_parameters():
    generator = Generator(100)
    discriminator = Discriminator()
    images = generator(torch.randn(4, 100))
    assert images.shape == (4, 1, 28, 28)
    assert discriminator(images).shape == (4, 1)
    assert parameter_count(generator) > 0 and parameter_count(discriminator) > 0


def test_losses_are_finite_and_differentiable():
    real = torch.randn(5, 1, requires_grad=True)
    fake = torch.randn(5, 1, requires_grad=True)
    d_loss = discriminator_loss(real, fake)
    g_loss = generator_loss(fake)
    assert torch.isfinite(d_loss) and torch.isfinite(g_loss)
    g_loss.backward()
    assert fake.grad is not None


def test_tiny_training_updates_weights_and_saves_artifacts(tmp_path):
    torch.manual_seed(10)
    images = torch.rand(8, 1, 28, 28) * 2 - 1
    labels = torch.zeros(8, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
    config = TrainConfig(category=0, epochs=1, batch_size=4, latent_dim=8, sample_count=4)
    result = train(config, loader, tmp_path, torch.device("cpu"))
    assert result["status"] == "completado"
    assert len(result["history"]) == 1
    assert (tmp_path / "checkpoint.pt").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "samples_epoch_001.pt").exists()
    initial = torch.load(tmp_path / "initial_weights.pt", map_location="cpu", weights_only=True)
    changed = any(not torch.equal(value, initial["generator"][key])
                  for key, value in result["generator"].state_dict().items())
    assert changed


def test_checkpoint_round_trip(tmp_path):
    generator, discriminator = create_models(8, torch.device("cpu"))
    from torch.optim import Adam
    g_opt, d_opt = Adam(generator.parameters()), Adam(discriminator.parameters())
    from training.trainer import save_checkpoint, load_checkpoint
    path = tmp_path / "roundtrip.pt"
    save_checkpoint(path, generator, discriminator, g_opt, d_opt, {"latent_dim": 8}, 2, [], torch.randn(3, 8))
    generator2, discriminator2 = create_models(8, torch.device("cpu"))
    payload = load_checkpoint(path, generator2, discriminator2, map_location="cpu")
    assert payload["epoch"] == 2
    for key, value in generator.state_dict().items():
        assert torch.equal(value, generator2.state_dict()[key])
