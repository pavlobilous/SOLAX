"""
The class registry backing solax's save/load machinery.

"dictify"/"undictify" (see solax.save_load.dictification) only know how
to convert a Python object to/from a plain dict if the object's class
has been registered here beforehand: each registered class is bound to
a string "label" (used inside the saved dict/JSON as the ".class" tag)
and a reconstruction callable that rebuilds an instance from its
dictified attributes. solax's own classes (Basis, State, OperatorTerm,
Operator, OperatorMatrix, RandomKeys) register themselves as a side
effect of being imported; user classes can be registered the same way
via the shared "save_load_registry" instance defined at the bottom of
this module.
"""
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
    """
    A label <-> (class, reconstruction callable) registry used by
    dictify()/undictify() to serialize/deserialize registered classes.

    Each entry maps a string "label" to a pair (cls, init_from_attr):
    "cls" is the registered class itself (used to look up its label
    again from an instance, via retreive_label()), and
    "init_from_attr" is a callable that, given the class's dictified
    "__dict__" attributes as keyword arguments, returns a reconstructed
    instance (commonly just the class itself, if its "__init__" accepts
    the same attributes as keyword arguments; otherwise a dedicated
    factory function).

    The module-level "save_load_registry" is the single shared instance
    used throughout solax; tests may use a private instance (e.g. via
    the "clean_registry" fixture) to register throwaway classes without
    polluting the shared one.
    """
    registry: RegType[Any] = field(default_factory=dict)

    def register(self, label: str, cls: Cls, init_from_attr: InitFromAttr[Cls]):
        """
        Registers a class for dictification.

        Input:
            - "label": the string tag under which "cls" will be saved
                (stored as the ".class" value in the dictified/JSON form);
                must not already be in use.
            - "cls": the class being registered.
            - "init_from_attr": a callable that reconstructs an instance
                of "cls" from its dictified "__dict__" attributes, passed
                as keyword arguments (see undictify()).
        Raises:
            RuntimeError if "label" is already registered.
        """
        if label in self.registry:
            raise RuntimeError(f"Label {label} is already in the registry. "\
                                "Choose another label or unregister this class")
        self.registry[label] = (cls, init_from_attr)


    def unregister(self, label: str):
        """
        Removes the class registered under "label".
        Raises KeyError if "label" is not currently registered.
        """
        try:
            del self.registry[label]
        except KeyError:
            raise KeyError(f"The label {label} not found in the registry.") from None


    def list_registered(self, ):
        """
        Returns a list of all currently registered labels.
        """
        return list(self.registry.keys())


    def retreive_init(self, label: str) -> InitFromAttr[Any]:
        """
        Returns the reconstruction callable registered under "label".
        Raises KeyError if "label" is not currently registered.
        """
        try:
            return self.registry[label][1]
        except KeyError:
            raise KeyError(f"Label {label} not found in the registry.") from None


    def retreive_cls(self, label: str) -> Any:
        """
        Returns the class registered under "label".
        Raises KeyError if "label" is not currently registered.
        """
        try:
            return self.registry[label][0]
        except KeyError:
            raise KeyError(f"Label {label} not found in the registry.") from None


    def retreive_label(self, cls: Any) -> str:
        """
        Returns the label a given class "cls" was registered under
        (found by identity, i. e. "cls" itself, not a subclass or an
        instance, must have been the object passed to register()).
        Raises TypeError if "cls" is not currently registered.
        """
        for k, (v, _) in self.registry.items():
            if v is cls:
                return k
        raise TypeError(f"Class {cls} not found in the registry.")


    def purge(self):
        """
        Removes all entries from the registry, leaving it empty.
        """
        self.registry.clear()


save_load_registry = SaveLoadRegistry()
"""The single SaveLoadRegistry instance shared across solax; all of
solax's own saveable classes register themselves here on import, and
"solax.save_load.save"/"solax.save_load.load" look classes up here."""