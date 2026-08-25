import jax
import pytest

from solax.save_load.registration import save_load_registry


def pytest_collection_modifyitems(config, items):
    if jax.device_count() >= 2:
        return
    skip_multi_device = pytest.mark.skip(
        reason="needs >=2 local JAX devices, e.g. "
        "XLA_FLAGS=--xla_force_host_platform_device_count=2"
    )
    for item in items:
        if "multi_device" in item.keywords:
            item.add_marker(skip_multi_device)


@pytest.fixture
def clean_registry():
    """Snapshot save_load_registry and unregister anything a test adds.

    save_load_registry is a process-wide singleton (solax's built-in classes
    register themselves at import time and must stay registered), so tests
    that register their own throwaway classes must clean up after themselves
    to stay isolated from one another.
    """
    before = set(save_load_registry.registry.keys())
    yield save_load_registry
    after = set(save_load_registry.registry.keys())
    for label in after - before:
        save_load_registry.unregister(label)
