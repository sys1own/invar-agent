"""Composable deterministic scientific instrument emitters."""
from . import calibration, executable_paper, hybrid_compute, receipts, workbench
from .base import EmitterRegistry, ModuleEmission


def default_emitters():
    registry = EmitterRegistry()
    registry.register('kernel', lambda node, context: ModuleEmission())
    for name, module in (('hybrid_compute', hybrid_compute), ('workbench', workbench), ('executable_paper', executable_paper), ('calibration', calibration), ('receipts', receipts)):
        registry.register(name, module.emit)
    return registry
