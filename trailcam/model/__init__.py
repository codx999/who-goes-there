"""Models: the small CNN from scratch, the pretrained backbones and the `build_model` factory."""

from trailcam.model.factory import MODEL_NAMES, ParameterGroups, build_model, parameter_groups

__all__ = ["MODEL_NAMES", "ParameterGroups", "build_model", "parameter_groups"]
