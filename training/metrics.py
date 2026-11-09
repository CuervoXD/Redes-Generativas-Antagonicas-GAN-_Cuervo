"""Esquema y utilidades de métricas por época."""
METRIC_COLUMNS = ["epoch", "generator_loss", "discriminator_loss", "elapsed_seconds",
                  "images_processed", "real_probability", "fake_probability"]


def make_epoch_record(epoch, g_loss, d_loss, elapsed, images, real_probability, fake_probability):
    """Construye una fila con las mismas columnas en toda ejecución."""
    return {"epoch": int(epoch), "generator_loss": float(g_loss),
            "discriminator_loss": float(d_loss), "elapsed_seconds": float(elapsed),
            "images_processed": int(images), "real_probability": float(real_probability),
            "fake_probability": float(fake_probability)}
