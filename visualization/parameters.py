"""Inspección cuantitativa de pesos y sesgos reales."""
import torch
from torch import nn


def layer_parameter_report(model):
    """Devuelve por capa formas, número de parámetros y estadísticas tensoriales."""
    reports = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        weights = module.weight.detach().float().cpu()
        bias = module.bias.detach().float().cpu() if module.bias is not None else None
        reports.append({"name": name, "type": type(module).__name__,
                        "input_features": module.in_features, "output_features": module.out_features,
                        "parameters": sum(p.numel() for p in module.parameters()),
                        "weight_shape": tuple(weights.shape),
                        "weight_stats": {"min": weights.min().item(), "max": weights.max().item(),
                                         "mean": weights.mean().item(), "std": weights.std(unbiased=False).item()},
                        "bias_shape": tuple(bias.shape) if bias is not None else None,
                        "weight_sample": weights.flatten()[:12].tolist(),
                        "bias_sample": bias.flatten()[:12].tolist() if bias is not None else []})
    return reports


def parameter_count(model):
    """Cuenta parámetros entrenables de una red."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def state_dict_difference(initial, current):
    """Mide si los tensores actuales se separaron de la inicialización guardada."""
    result = {}
    for key, tensor in current.items():
        if key in initial:
            result[key] = float((tensor.detach().cpu() - initial[key].detach().cpu()).abs().mean())
    return result
