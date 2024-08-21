from collections.abc import Callable
from typing import Any, TypeVar
from dataclasses import dataclass, field


Cls = TypeVar("Cls")

InitFromAttr = Callable[..., Cls]
RegValueType = tuple[Cls, InitFromAttr[Cls]]
RegType = dict[str, RegValueType[Cls]]


registry: RegType[Any] = {}


@dataclass
class SaveLoadRegistry:
    registry: RegType[Any] = field(default_factory=dict)

    def register(self, label: str, cls: Cls, init_from_attr: InitFromAttr[Cls]):
        """
        Registers a class for dictification.
        """
        if label in self.registry:
            raise RuntimeError(f"Label {label} is already in the registry. "\
                                "Choose another label or unregister this class")
        self.registry[label] = (cls, init_from_attr)


    def unregister(self, label: str):
        try:
            del self.registry[label]
        except KeyError:
            raise KeyError(f"The label {label} not found in the registry.")


    def list_registered(self, ):
        return list(self.registry.keys())


    def retreive_init(self, label: str) -> InitFromAttr[Any]:
        try:
            return self.registry[label][1]
        except KeyError:
            raise KeyError(f"Label {label} not found in the registry.")


    def retreive_cls(self, label: str) -> Any:
        try:
            return self.registry[label][0]
        except KeyError:
            raise KeyError(f"Label {label} not found in the registry.")


    def retreive_label(self, cls: Any) -> str:
        for k, (v, _) in self.registry.items():
            if v is cls:
                return k
        raise TypeError(f"Class {cls} not found in the registry.")


    def purge(self):
        self.registry.clear()


save_load_registry = SaveLoadRegistry()