"""Composable deterministic scientific instrument emitters."""
from . import calibration, executable_paper, hybrid_compute, receipts, workbench
from .base import EmitterRegistry, ModuleEmission


def _empty_kernel_emit(node, context):
    return ModuleEmission()


def default_emitters():
    registry = EmitterRegistry()
    registry.process_factory = "invar_agent.builder.emitters:default_emitters"
    registry.register('kernel', _empty_kernel_emit)
    for name, module in (('hybrid_compute', hybrid_compute), ('workbench', workbench), ('executable_paper', executable_paper), ('calibration', calibration), ('receipts', receipts)):
        registry.register(name, module.emit)
    return registry
