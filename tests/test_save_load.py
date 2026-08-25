"""
Pytest port of _pjax-master-tests/tests/3.save_load.ipynb.

Covers solax's save/load machinery in three layers, mirroring the notebook's
own section structure:
    1. solax.save_load.json_for_dicts: dump_dict_with_nd / load_dict_with_nd
       -- low-level (de)serialization of nested dicts that may contain
       NumPy arrays.
    2. solax.save_load.dictification: dictify / undictify -- translating
       (registered) Python/dataclass objects to/from plain dicts.
    3. solax.save_load (save / load) -- the user-facing save/load API that
       combines the two layers above and is used for real SOLAX objects
       (Basis, Operator, RandomKeys, ...).

Notebook `help(...)` cells and shell-magic `!ls` / `rm -r` cells are dropped
entirely (see task conventions); their intent -- checking that files exist on
disk, and cleaning up afterwards -- is instead handled with pytest's
`tmp_path` fixture. Object reprs / numpy scalar reprs are never asserted on
verbatim; only real attributes/values/types are checked.
"""

import os

import numpy as np
import pytest
from dataclasses import dataclass

import solax as sx
from solax.random_keys import RandomKeys
from solax.save_load import save, load
from solax.save_load.dictification import dictify, undictify
from solax.save_load.json_for_dicts import dump_dict_with_nd, load_dict_with_nd
from solax.save_load.registration import save_load_registry


# ---------------------------------------------------------------------------
# 1. Save / load nested dicts with NumPy arrays (json_for_dicts)
# ---------------------------------------------------------------------------

def _example_nested_dict():
    return {
        "here": "my_data",
        "which": ["will", "be", "json-ed."],
        "BUT": None,
        "if": {"a": "NumPy array", "is": "encountered", "it_will_be": "saved."},
        "check_this_out": np.arange(3),
        "or_nested_version": {
            "one": np.zeros((3, 3)),
            "three": np.arange(12).reshape(3, 4),
            "four": np.arange(6).sum(),
        },
    }


def test_dump_and_load_dict_with_nd_round_trips(tmp_path):
    d = _example_nested_dict()
    path = str(tmp_path / "saved_")

    dump_dict_with_nd(d, path)
    loaded = load_dict_with_nd(path)

    assert loaded["here"] == "my_data"
    assert loaded["which"] == ["will", "be", "json-ed."]
    assert loaded["BUT"] is None
    assert loaded["if"] == {
        "a": "NumPy array", "is": "encountered", "it_will_be": "saved.",
    }
    np.testing.assert_array_equal(loaded["check_this_out"], np.arange(3))
    np.testing.assert_array_equal(loaded["or_nested_version"]["one"], np.zeros((3, 3)))
    np.testing.assert_array_equal(
        loaded["or_nested_version"]["three"], np.arange(12).reshape(3, 4)
    )
    # "four" was a 0-d numpy scalar (np.arange(6).sum()); it round-trips
    # through .npy as a 0-d ndarray rather than a bare Python int.
    np.testing.assert_array_equal(loaded["or_nested_version"]["four"], 15)


def test_dump_dict_with_nd_saves_one_npy_file_per_array(tmp_path):
    d = _example_nested_dict()
    path = tmp_path / "saved_"

    dump_dict_with_nd(d, str(path))

    assert (path / "schema.json").exists()
    assert (path / "check_this_out.npy").exists()
    assert (path / "or_nested_version" / "one.npy").exists()
    assert (path / "or_nested_version" / "three.npy").exists()
    assert (path / "or_nested_version" / "four.npy").exists()


def test_dump_dict_with_nd_rejects_non_dict_top_level(tmp_path):
    path = str(tmp_path / "saved_")

    with pytest.raises(TypeError, match='"nested_dict" must be a dict.'):
        dump_dict_with_nd(123, path)

    with pytest.raises(TypeError, match='"nested_dict" must be a dict.'):
        dump_dict_with_nd(np.arange(4), path)


# ---------------------------------------------------------------------------
# 2. Dictification / undictification of classes
# ---------------------------------------------------------------------------

@dataclass
class MyClass1:
    field1: int
    field2: str
    field3: dict


@dataclass
class MyClass2:
    data: MyClass1


def test_registration_adds_and_lists_labels(clean_registry):
    before = set(clean_registry.list_registered())

    clean_registry.register("MyClass1", MyClass1, MyClass1)
    clean_registry.register("MyClass2", MyClass2, MyClass2)

    after = set(clean_registry.list_registered())
    assert after - before == {"MyClass1", "MyClass2"}
    # solax's own built-in classes must remain registered throughout.
    assert {"Basis", "State", "OperatorTerm", "OperatorMatrix",
            "Operator", "RandomKeys"} <= before


def test_registering_duplicate_label_raises(clean_registry):
    clean_registry.register("MyClass1", MyClass1, MyClass1)
    with pytest.raises(RuntimeError):
        clean_registry.register("MyClass1", MyClass1, MyClass1)


def test_unregister_unknown_label_raises_key_error(clean_registry):
    with pytest.raises(KeyError):
        clean_registry.unregister("NotRegisteredAtAll")


def test_dictify_undictify_single_class_round_trips(clean_registry):
    clean_registry.register("MyClass1", MyClass1, MyClass1)

    my_obj1 = MyClass1(1, "lala", {"blah_blah": [[1, 2, 3], np.arange(5)]})
    d1 = dictify(my_obj1)

    assert d1[".class"] == "MyClass1"
    assert d1[".attrs"]["field1"] == 1
    assert d1[".attrs"]["field2"] == "lala"

    restored = undictify(d1)
    assert isinstance(restored, MyClass1)
    assert restored.field1 == 1
    assert restored.field2 == "lala"
    assert restored.field3["blah_blah"][0] == [1, 2, 3]
    np.testing.assert_array_equal(restored.field3["blah_blah"][1], np.arange(5))


def test_dictify_undictify_nested_classes_in_dicts_round_trips(clean_registry):
    clean_registry.register("MyClass1", MyClass1, MyClass1)
    clean_registry.register("MyClass2", MyClass2, MyClass2)

    my_obj1 = MyClass1(1, "lala", {"blah_blah": [[1, 2, 3], np.arange(5)]})
    my_obj2 = MyClass2(my_obj1)
    nested_dict = dict(obj1=my_obj1, obj2=my_obj2)

    d12 = dictify(nested_dict)
    assert d12["obj1"][".class"] == "MyClass1"
    assert d12["obj2"][".class"] == "MyClass2"
    assert d12["obj2"][".attrs"]["data"][".class"] == "MyClass1"

    restored = undictify(d12)
    assert isinstance(restored["obj1"], MyClass1)
    assert isinstance(restored["obj2"], MyClass2)
    assert isinstance(restored["obj2"].data, MyClass1)
    assert restored["obj1"].field1 == 1
    assert restored["obj2"].data.field2 == "lala"


# --- classes inheriting from dict, and classes with their own __save__ ----

@dataclass
class MyDict(dict):
    a: float


@dataclass
class MyDictClass(MyDict):
    b: int
    c: np.ndarray


@dataclass
class MyDictClassWithSL(MyDict):
    b: int
    c: np.ndarray

    def __save__(self):
        pass

    def __load__(self):
        pass


def test_dict_subclass_dictify_undictify_round_trips_attrs_and_dict_items(clean_registry):
    clean_registry.register("MyDictClass", MyDictClass, MyDictClass)

    mdc = MyDictClass(1.23, 5, np.arange(3))
    mdc["key1"] = "label1"
    mdc["key2"] = "label2"
    assert dict(mdc) == {"key1": "label1", "key2": "label2"}

    d = dictify(mdc)
    assert d[".class"] == "MyDictClass"
    assert d[".attrs"]["a"] == 1.23
    assert d[".attrs"]["b"] == 5
    np.testing.assert_array_equal(d[".attrs"]["c"], np.arange(3))
    assert d[".dict"] == {"key1": "label1", "key2": "label2"}

    ud = undictify(d)
    assert isinstance(ud, MyDictClass)
    assert ud.a == 1.23
    assert ud.b == 5
    np.testing.assert_array_equal(ud.c, np.arange(3))
    assert dict(ud) == {"key1": "label1", "key2": "label2"}


def test_class_with_own_save_load_is_kept_as_class_not_instance(clean_registry):
    """A class that defines __save__/__load__ opts out of dictification:
    dictify() records only its label, and undictify() hands back the class
    itself (not a reconstructed instance) -- solax leaves it to the object's
    own __save__/__load__ machinery instead.
    """
    clean_registry.register("MyDictClassWithSL", MyDictClassWithSL, MyDictClassWithSL)

    mdc_withsl = MyDictClassWithSL(1.23, 5, np.arange(3))
    d_withsl = dictify(mdc_withsl)

    assert d_withsl == {".class_with_own_svld": "MyDictClassWithSL"}

    ud_withsl = undictify(d_withsl)
    assert ud_withsl is MyDictClassWithSL


def test_mixed_dict_of_plain_and_own_save_load_classes(clean_registry):
    clean_registry.register("MyDictClass", MyDictClass, MyDictClass)
    clean_registry.register("MyDictClassWithSL", MyDictClassWithSL, MyDictClassWithSL)

    mdc = MyDictClass(1.23, 5, np.arange(3))
    mdc["key1"] = "label1"
    mdc["key2"] = "label2"
    mdc_withsl = MyDictClassWithSL(1.23, 5, np.arange(3))

    cls12 = dict(mdc=mdc, mdc_withsl=mdc_withsl)
    d12 = dictify(cls12)

    assert d12["mdc"][".class"] == "MyDictClass"
    assert d12["mdc_withsl"] == {".class_with_own_svld": "MyDictClassWithSL"}

    restored = undictify(d12)
    assert isinstance(restored["mdc"], MyDictClass)
    assert dict(restored["mdc"]) == {"key1": "label1", "key2": "label2"}
    assert restored["mdc_withsl"] is MyDictClassWithSL


@dataclass
class MyDictSuperClass(dict):
    mdc: "MyDictClass"


def test_nested_dict_subclasses_round_trip(clean_registry):
    clean_registry.register("MyDictClass", MyDictClass, MyDictClass)
    clean_registry.register("MyDictSuperClass", MyDictSuperClass, MyDictSuperClass)

    mdc = MyDictClass(1.23, 5, np.arange(3))
    mdc["key1"] = "label1"
    mdc["key2"] = "label2"

    mdsc = MyDictSuperClass(mdc)
    mdsc["super_data"] = 123
    mdsc["shallow_copy"] = mdc

    mdsc_dict = dictify(mdsc)
    assert mdsc_dict[".class"] == "MyDictSuperClass"
    assert mdsc_dict[".attrs"]["mdc"][".class"] == "MyDictClass"
    assert mdsc_dict[".dict"]["super_data"] == 123
    assert mdsc_dict[".dict"]["shallow_copy"][".class"] == "MyDictClass"

    mdsc_undict = undictify(mdsc_dict)
    assert isinstance(mdsc_undict.mdc, MyDictClass)
    assert mdsc_undict.mdc.a == 1.23
    restored_dict_items = dict(mdsc_undict)
    assert restored_dict_items["super_data"] == 123
    assert isinstance(restored_dict_items["shallow_copy"], MyDictClass)
    np.testing.assert_array_equal(restored_dict_items["shallow_copy"].c, np.arange(3))


# --- classes needing __pre_dictify__ / __post_undictify__ -----------------

class MyClassWithPrePostNeeded:
    def __init__(self, a, b, c, d):
        self.data = {(0, 0): a, (0, 1): b, (1, 0): c, (1, 1): d}


def test_dictify_rejects_non_string_dict_keys_without_pre_dictify(clean_registry):
    clean_registry.register(
        "MyClassWithPrePostNeeded", MyClassWithPrePostNeeded, lambda: None
    )
    mc_pp_needed = MyClassWithPrePostNeeded(5, 6, 7, 8)

    with pytest.raises(TypeError, match="All dict keys must be strings."):
        dictify(mc_pp_needed)


class MyClassWithPrePost:
    def __init__(self, a, b, c, d):
        self.data = {(0, 0): a, (0, 1): b, (1, 0): c, (1, 1): d}

    def __pre_dictify__(self):
        tpl_to_str = lambda tpl: "".join(str(v) for v in tpl)
        pseudo_obj = MyClassWithPrePost(0, 0, 0, 0)
        pseudo_obj.data = {"_" + tpl_to_str(k): v for k, v in self.data.items()}
        return pseudo_obj

    def __post_undictify__(self):
        str_to_tpl = lambda s: tuple(int(v) for v in s[1:])
        obj = MyClassWithPrePost(0, 0, 0, 0)
        obj.data = {str_to_tpl(k): v for k, v in self.data.items()}
        return obj


def _init_myclasswithprepost_from_attr(data):
    obj = MyClassWithPrePost(0, 0, 0, 0)
    obj.data = data
    return obj


def test_pre_dictify_and_post_undictify_round_trip_tuple_keys(clean_registry):
    clean_registry.register(
        "MyClassWithPrePost", MyClassWithPrePost, _init_myclasswithprepost_from_attr
    )

    mc_pp = MyClassWithPrePost(9, 10, 11, 12)
    assert mc_pp.data == {(0, 0): 9, (0, 1): 10, (1, 0): 11, (1, 1): 12}

    mc_pp_dict = dictify(mc_pp)
    assert mc_pp_dict == {
        ".class": "MyClassWithPrePost",
        ".attrs": {"data": {"_00": 9, "_01": 10, "_10": 11, "_11": 12}},
    }

    obj_undict = undictify(mc_pp_dict)
    assert isinstance(obj_undict, MyClassWithPrePost)
    assert obj_undict.data == {(0, 0): 9, (0, 1): 10, (1, 0): 11, (1, 1): 12}


# ---------------------------------------------------------------------------
# 3. Saving / loading nested (SOLAX) classes in dicts
# ---------------------------------------------------------------------------

def _example_basis():
    return sx.Basis("00 11 01 10 00 11".split())


def test_save_load_single_basis_round_trips(tmp_path):
    basis = _example_basis()
    path = str(tmp_path / "saved_basis")

    save(basis, path)
    assert os.path.exists(os.path.join(path, ".attrs", "_encoding.npy"))

    loaded = load(path)
    assert isinstance(loaded, sx.Basis)
    assert loaded == basis


def test_save_load_dict_of_many_bases_and_info(tmp_path):
    basis = _example_basis()
    path = str(tmp_path / "saved_many")

    dct_to_save = dict(basis1=basis, basis2=basis, info="some info")
    save(dct_to_save, path)

    assert os.path.exists(os.path.join(path, "schema.json"))
    assert os.path.isdir(os.path.join(path, "basis1"))
    assert os.path.isdir(os.path.join(path, "basis2"))

    dct_loaded = load(path)
    assert dct_loaded["basis1"] == basis
    assert dct_loaded["basis2"] == basis
    assert dct_loaded["info"] == "some info"


def test_save_load_single_basis_via_generic_path(tmp_path):
    o1 = _example_basis()
    path = str(tmp_path / "saved_o1")

    save(o1, path)
    l1 = load(path)
    assert l1 == o1


def test_save_load_flat_dict_with_two_keys_and_info(tmp_path):
    o1 = _example_basis()
    o2 = _example_basis()
    path = str(tmp_path / "saved_flat")

    d = dict(key1=o1, key2=o2, info="some info")
    save(d, path)
    l2 = load(path)

    assert l2["key1"] == o1
    assert l2["key2"] == o2
    assert l2["info"] == "some info"


def test_save_load_nested_dict_with_basis_and_extra_ndarray(tmp_path):
    o1 = _example_basis()
    o2 = _example_basis()
    path = str(tmp_path / "saved_nested")

    d = dict(
        key1=o1,
        key=dict(key2=o2, extra_data=np.array([1.23, 4.56])),
        info="some info",
    )
    save(d, path)
    l3 = load(path)

    assert l3["key1"] == o1
    assert l3["key"]["key2"] == o2
    np.testing.assert_array_equal(l3["key"]["extra_data"], np.array([1.23, 4.56]))
    assert l3["info"] == "some info"


def test_save_load_empty_dict_round_trips(tmp_path):
    path = str(tmp_path / "saved_empty")

    save({}, path)
    assert os.path.exists(os.path.join(path, "schema.json"))

    loaded = load(path)
    assert loaded == {}


def test_save_twice_to_same_path_overwrites_cleanly(tmp_path):
    """save()/load() treat `path` as a directory that gets shutil.rmtree'd
    and recreated on a repeat save -- saving twice to the same path must
    still fully succeed and round-trip correctly (not merge with, or choke
    on, the previous contents).
    """
    path = str(tmp_path / "saved_twice")
    basis = _example_basis()
    other_basis = sx.Basis("00 01".split())

    save(basis, path)
    first_load = load(path)
    assert first_load == basis

    # Save something different to the very same path.
    save(other_basis, path)
    second_load = load(path)
    assert second_load == other_basis
    assert not (second_load == basis)

    # Directory must not have leftover files from the first save.
    assert sorted(os.listdir(os.path.join(path, ".attrs"))) == ["_encoding.npy"]


def test_save_load_dict_mixing_basis_and_operator(tmp_path):
    basis = _example_basis()
    op = sx.Operator((1, 0), np.array([[0, 1]]), np.array([1.0]))
    path = str(tmp_path / "saved_mixed")

    save(dict(basis=basis, op=op), path)
    loaded = load(path)

    assert loaded["basis"] == basis
    loaded_op = loaded["op"]
    assert isinstance(loaded_op, sx.Operator)
    assert set(loaded_op.keys()) == set(op.keys())
    for key in op.keys():
        term, loaded_term = op[key], loaded_op[key]
        assert loaded_term.daggers == term.daggers
        np.testing.assert_array_equal(loaded_term.posits, term.posits)
        np.testing.assert_array_equal(loaded_term.coeffs, term.coeffs)


def test_save_load_deeply_nested_dict_with_random_keys(tmp_path):
    """Deep nesting mixing: plain values, a nested dict of ints, a Basis,
    a dict of two NumPy arrays, and a RandomKeys instance (which relies on
    its own __pre_dictify__/__post_undictify__ to survive the JAX <-> NumPy
    boundary).
    """
    basis = _example_basis()
    rk = RandomKeys(seed=42)
    path = str(tmp_path / "saved_deep")

    deep = dict(
        plain=dict(x=1, y="hello", z=None),
        counts=dict(a=1, b=2, c=3),
        basis=basis,
        arrays=dict(arr1=np.arange(4), arr2=np.eye(2)),
        rk=rk,
    )
    save(deep, path)
    loaded = load(path)

    assert loaded["plain"] == {"x": 1, "y": "hello", "z": None}
    assert loaded["counts"] == {"a": 1, "b": 2, "c": 3}
    assert loaded["basis"] == basis
    np.testing.assert_array_equal(loaded["arrays"]["arr1"], np.arange(4))
    np.testing.assert_array_equal(loaded["arrays"]["arr2"], np.eye(2))

    loaded_rk = loaded["rk"]
    assert isinstance(loaded_rk, RandomKeys)
    np.testing.assert_array_equal(np.array(loaded_rk._key), np.array(rk._key))


# --- error paths of save() -------------------------------------------------

def test_save_rejects_solax_object_nested_in_a_plain_list(tmp_path):
    """Each SOLAX object to be saved must have an associated dict key --
    burying it inside a plain list under a key is not supported.
    """
    basis = _example_basis()
    path = str(tmp_path / "saved_bad")
    sv = {"a": [basis, "info"]}

    with pytest.raises(TypeError, match="Something wrong passed to the saver."):
        save(sv, path)


def test_save_rejects_bare_non_dict_non_solax_value(tmp_path):
    path = str(tmp_path / "saved_bad2")

    with pytest.raises(TypeError, match="Something wrong passed to the saver."):
        save(1, path)


def test_save_rejects_unregistered_class_instance(clean_registry, tmp_path):
    """my_obj1 here is a plain (unregistered) object -- solax's save() can
    only handle registered SOLAX classes or dicts.
    """
    path = str(tmp_path / "saved_bad3")

    @dataclass
    class NotRegistered:
        x: int

    with pytest.raises(TypeError, match="Something wrong passed to the saver."):
        save(NotRegistered(1), path)


def test_save_rejects_non_string_dict_key(tmp_path):
    path = str(tmp_path / "saved_bad4")

    with pytest.raises(TypeError, match="Something wrong passed to the saver."):
        save({1: (lambda a: a)}, path)


def test_save_rejects_non_identifier_string_dict_key(tmp_path):
    """Unlike the other error paths above (which all go through dictify's
    TypeError and get converted by save() into its own generic message),
    a syntactically invalid-but-string key hits assert_valid_key's
    ValueError branch -- and save() only catches TypeError, so this
    ValueError propagates to the caller unconverted, with its own message.
    This is a genuine (if perhaps unintentional) asymmetry in
    solax/save_load/usr_save_load.py's error handling; it's exercised here
    rather than worked around.
    """
    path = str(tmp_path / "saved_bad5")

    with pytest.raises(ValueError, match="All dict keys must be valid variable identifiers."):
        save({"1abc": 5}, path)
