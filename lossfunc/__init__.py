from functools import partial
from typing import *

from deepclustering2.utils.general import _register
from torch import nn

from networks.enet import Enet

from networks.vgg import VGG16

__all__ = [
    "get_arch",
    "ARCH_CALLABLES",
    "_register_arch",
]

"""
Package
"""
# A Map from string to arch callables
ARCH_CALLABLES: Dict[str, Callable] = {}
_register_arch = partial(_register, CALLABLE_DICT=ARCH_CALLABLES)


"""
Public interface
"""
_register_arch("enet", Enet)
_register_arch("vgg16", VGG16)


def get_arch(arch: str, kwargs) -> nn.Module:
    """ Get the architecture. Return a torch.nn.Module """
    arch_callable = ARCH_CALLABLES.get(arch.lower())
    kwargs.pop("arch", None)
    assert arch_callable, "Architecture {} is not found!".format(arch)
    net = arch_callable(**kwargs)
    return net
