"""
End-to-end tests of solax's save/load subsystem from a *custom-class
developer's* point of view: each test below plays out one of the three
ways described in solax.save_load.dictification to make a class of
one's own savable/loadable, exercised through the public save()/load()
API (not dictify()/undictify() directly, which tests/test_save_load.py
already covers) -- i.e. real files under a real tmp_path.

1. The plain path: attributes are already solax/NumPy/primitive
   objects, so registering the class (with its own constructor as the
   reconstruction callable) is enough.
2. __pre_dictify__/__post_undictify__: for a class holding something
   that isn't directly dict/JSON-safe (here, a dict keyed by tuples).
3. __save__: opting a class out of dictification entirely -- solax
   then does not persist or restore its state at all; that is left
   entirely to the developer's own means, outside solax.save()/load().
"""
import numpy as np
import pytest

from solax.save_load import save, load


# ---------------------------------------------------------------------------
# 1. Plain registration: attributes are already solax/NumPy/primitive
# ---------------------------------------------------------------------------

class SensorCalibration:
    """A fake small custom class: per-channel calibration offsets plus
    a single gain factor. Its attributes (a string, a NumPy array, a
    float) are all directly dictifiable, so no __pre_dictify__/
    __post_undictify__ is needed -- registering the class itself as
    its own reconstruction callable is enough, since __init__ already
    takes its attributes by name.
    """
    def __init__(self, name, offsets, gain):
        self.name = name
        self.offsets = offsets
        self.gain = gain


def test_plain_custom_class_save_load_round_trips(clean_registry, tmp_path):
    clean_registry.register("SensorCalibration", SensorCalibration, SensorCalibration)

    original = SensorCalibration("temp_probe", np.array([0.1, -0.2, 0.05]), gain=2.5)
    path = str(tmp_path / "saved_sensor")

    save(original, path)
    loaded = load(path)

    assert isinstance(loaded, SensorCalibration)
    assert loaded.name == "temp_probe"
    np.testing.assert_array_equal(loaded.offsets, original.offsets)
    assert loaded.gain == 2.5


# ---------------------------------------------------------------------------
# 2. __pre_dictify__ / __post_undictify__: an attribute that isn't
#    directly dict/JSON-safe (a dict keyed by (row, col) tuples)
# ---------------------------------------------------------------------------

class SparseGrid:
    """A fake small custom class: a sparse 2D grid, {(row, col): value}.
    Tuple keys aren't valid dict/JSON keys in solax's save/load (see
    assert_valid_key()), so this needs __pre_dictify__/__post_undictify__
    to present a string-keyed surrogate for saving and convert it back
    on loading -- the same pattern solax's own RandomKeys uses.
    """
    def __init__(self, points):
        self.points = dict(points)

    def __pre_dictify__(self):
        surrogate = SparseGrid({})
        surrogate.points = {f"p{r}_{c}": v for (r, c), v in self.points.items()}
        return surrogate

    def __post_undictify__(self):
        def key_from_str(s):
            r, c = s[1:].split("_")
            return int(r), int(c)
        restored = SparseGrid({})
        restored.points = {key_from_str(k): v for k, v in self.points.items()}
        return restored


def test_custom_class_with_pre_post_dictify_save_load_round_trips(clean_registry, tmp_path):
    clean_registry.register("SparseGrid", SparseGrid, SparseGrid)

    original = SparseGrid({(0, 0): 1.5, (1, 2): -3.0, (7, 3): 0.0})
    path = str(tmp_path / "saved_grid")

    save(original, path)
    loaded = load(path)

    assert isinstance(loaded, SparseGrid)
    assert loaded.points == original.points


# ---------------------------------------------------------------------------
# 3. __save__: opting out of dictification entirely
# ---------------------------------------------------------------------------

class OpaqueBlob:
    """A fake small custom class demonstrating the __save__ opt-out:
    defining __save__ tells solax not to dictify this class's
    attributes at all. solax.save() then persists only the class's
    label (none of "payload" below), and solax.load() hands back the
    class itself, not a reconstructed instance -- actually
    saving/restoring an OpaqueBlob's state is left entirely to the
    developer, by whatever means they choose, outside solax.save()/
    solax.load().
    """
    def __init__(self, payload):
        self.payload = payload

    def __save__(self):
        pass


def test_custom_class_with_save_hook_is_not_persisted_by_solax(clean_registry, tmp_path):
    clean_registry.register("OpaqueBlob", OpaqueBlob, OpaqueBlob)

    original = OpaqueBlob(payload=np.arange(5))
    path = tmp_path / "saved_blob"

    save(original, str(path))

    # Only the class label was written -- "payload" was never persisted.
    assert (path / "schema.json").read_text().strip() == '{".class_with_own_svld": "OpaqueBlob"}'

    loaded = load(str(path))
    assert loaded is OpaqueBlob
    with pytest.raises(AttributeError):
        _ = loaded.payload  # no instance was (or could be) reconstructed


# ---------------------------------------------------------------------------
# 4. All three, plus a built-in solax class, composed in one save
# ---------------------------------------------------------------------------

def test_custom_classes_compose_with_each_other_and_builtin_classes(clean_registry, tmp_path):
    import solax as sx

    clean_registry.register("SensorCalibration", SensorCalibration, SensorCalibration)
    clean_registry.register("SparseGrid", SparseGrid, SparseGrid)
    clean_registry.register("OpaqueBlob", OpaqueBlob, OpaqueBlob)

    sensor = SensorCalibration("temp_probe", np.array([0.1, -0.2]), gain=1.0)
    grid = SparseGrid({(0, 1): 2.0})
    blob = OpaqueBlob(payload="anything")
    basis = sx.Basis(["10", "01"])
    path = str(tmp_path / "saved_everything")

    save(dict(sensor=sensor, grid=grid, blob=blob, basis=basis), path)
    loaded = load(path)

    assert loaded["sensor"].name == "temp_probe"
    assert loaded["grid"].points == {(0, 1): 2.0}
    assert loaded["blob"] is OpaqueBlob
    assert loaded["basis"] == basis
