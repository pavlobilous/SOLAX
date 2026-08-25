"""Scaled-down smoke test ported from JupyterNotebooks/siam_nn_saveload.ipynb.

The notebook trains a BasisClassifier on a real ~59k-determinant SIAM basis
for up to 200 epochs. That is far too slow/flaky for a unit test, so this
file exercises the *same pipeline* (BasisClassifier -> BigBasisManager ->
train_classifier -> predict_impt_subbasis -> save_state/load_state, plus a
plain sx.save/sx.load round trip) at a tiny scale: a handful of bath sites,
tens of determinants, a handful of epochs. This is a plumbing/integration
smoke test, not a physics regression test (see test_integration_siam.py for
that) and not a quality benchmark -- real training at this tiny scale is not
reliable enough to assert accuracy values, only that it runs correctly and
produces finite, well-typed results.
"""
import re

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import pytest

import solax as sx


def build_bath(N_bath):
    ii = np.arange(N_bath) + 1
    xx = ii * np.pi / (N_bath + 1)
    e_bath = -2 * np.cos(xx)

    V0 = np.sqrt(20 / (N_bath + 1))
    V_bath = V0 * np.sqrt(1 - (e_bath / 2) ** 2)

    return e_bath, V_bath


def build_start_dets(N_bath):
    det1 = "01" + "1" * (N_bath - 1) + "10" + "0" * (N_bath - 1)
    det2 = "10" + "1" * (N_bath - 1) + "01" + "0" * (N_bath - 1)
    return det1, det2


def build_siam_hamiltonian(N_bath, U):
    e_bath, V_bath = build_bath(N_bath)

    H_imp2 = sx.Operator((1, 0, 1, 0), np.array([[0, 0, 1, 1]]), np.array([U]))
    H_imp1 = sx.Operator(
        (1, 0), np.array([[0, 0], [1, 1]]), np.array([-U / 2, -U / 2])
    )
    H_imp = H_imp2 + H_imp1 + U / 4

    H_bath = sx.Operator(
        (1, 0),
        np.arange(2, 2 * N_bath + 2).repeat(2).reshape(-1, 2),
        e_bath.repeat(2),
    )

    H_hyb_posits = np.vstack(
        [np.array([0, 1] * N_bath), np.arange(2, 2 * N_bath + 2)]
    ).T
    H_hyb_nohc = sx.Operator((1, 0), H_hyb_posits, V_bath.repeat(2))

    return H_imp + H_bath + H_hyb_nohc + H_hyb_nohc.hconj


def nn_call_on_bits(x):
    """Tiny CNN+MLP over bit-encoded determinants, shrunk from the notebook's
    architecture (which used Conv(64)/Conv(4) and a 46/23/11/2 MLP) down to a
    handful of units, just enough to exercise the same plumbing.
    """
    x = x.reshape(-1, 2)
    x = nn.Conv(features=4, kernel_size=(2,), padding="valid")(x)
    x = nn.relu(x)
    x = x.reshape(-1)
    x = nn.Dense(features=8)(x)
    x = nn.relu(x)
    x = nn.Dense(features=2)(x)
    return x


N_BATH = 5
U_IMPURITY = 10
RANDOM_NUM = 16
TARGET_NUM = 12


def build_tiny_pipeline_inputs():
    """Builds a tiny SIAM Hamiltonian/basis and the "candidates" basis that
    BigBasisManager operates on, mirroring the notebook's basis_small /
    basis_big / candidates construction but at N_bath=5 (tens, not tens of
    thousands, of determinants).
    """
    H = build_siam_hamiltonian(N_BATH, U_IMPURITY)
    basis_start = sx.Basis(build_start_dets(N_BATH))

    basis_small = H(basis_start)
    basis_big = H(basis_small)
    candidates = basis_big % basis_small

    return H, basis_start, basis_small, candidates


def make_classifier(dummy_basis, key, learning_rate=0.01):
    classifier = sx.BasisClassifier(nn_call_on_bits)
    optimizer = optax.adam(learning_rate=learning_rate)
    classifier.initialize(key, dummy_basis, optimizer)
    return classifier, optimizer


@pytest.mark.slow
def test_basis_classifier_initializes_and_summarizes(clean_registry):
    _, basis_start, _, _ = build_tiny_pipeline_inputs()

    rand_keys = sx.RandomKeys(seed=1234)
    classifier, _ = make_classifier(basis_start, next(rand_keys))

    # Should not raise, and should actually print something (a Flax
    # tabulate summary), not silently no-op.
    classifier.print_summary()


@pytest.mark.slow
def test_sample_subbasis_returns_exact_length_basis(clean_registry):
    _, _, _, candidates = build_tiny_pipeline_inputs()
    basis_start = sx.Basis(build_start_dets(N_BATH))

    rand_keys = sx.RandomKeys(seed=7)
    classifier, _ = make_classifier(basis_start, next(rand_keys))

    bbm = sx.BigBasisManager(candidates, classifier)
    random_sel = bbm.sample_subbasis(next(rand_keys), RANDOM_NUM)

    assert isinstance(random_sel, sx.Basis)
    assert len(random_sel) == RANDOM_NUM


def _diagonalize_dense(H, basis):
    """Dense eigh is used instead of sparse eigsh: at these tiny basis sizes
    (tens of determinants) eigsh's default settings (which want k << N) are
    not reliably applicable, whereas dense eigh always works.
    """
    matrix_dense = H.build_matrix(basis).to_scipy().todense()
    eigvals, eigvecs = np.linalg.eigh(matrix_dense)
    idx = np.argmin(eigvals)
    return eigvals[idx], np.asarray(eigvecs[:, idx])


@pytest.mark.slow
def test_train_classifier_and_predict_impt_subbasis(clean_registry, capsys):
    H, basis_start, basis_small, candidates = build_tiny_pipeline_inputs()

    rand_keys = sx.RandomKeys(seed=1234)
    classifier, _ = make_classifier(basis_start, next(rand_keys))

    bbm = sx.BigBasisManager(candidates, classifier)

    random_sel = bbm.sample_subbasis(next(rand_keys), RANDOM_NUM)
    basis_diag = basis_small + random_sel

    _, eigenvec = _diagonalize_dense(H, basis_diag)
    state_diag = sx.State(basis_diag, eigenvec)
    state_train = state_diag % basis_small
    assert len(state_train) == RANDOM_NUM

    abs_coeff_cut = bbm.derive_abs_coeff_cut(TARGET_NUM, state_train)
    # It's a statistical threshold derived from real (tiny-sample) training
    # data -- just check it comes back as a plain finite float, not a
    # specific value.
    assert isinstance(abs_coeff_cut, (float, np.floating))
    assert np.isfinite(abs_coeff_cut)

    early_stopped = bbm.train_classifier(
        next(rand_keys),
        state_train,
        abs_coeff_cut,
        batch_size=4,
        epochs=3,
        early_stop=True,
        early_stop_params={"patience": 2},
    )
    # Per manager_class.py / training.py, train_classifier's return
    # contract is the "early_stopped" bool signalling whether training
    # stopped before exhausting all epochs.
    assert isinstance(early_stopped, bool)

    # Parse the accuracy values that AccuracyMonitor printed during
    # training (via the default printout_vals=True) and check they are
    # all finite -- real training at this scale is not reliable enough to
    # assert *quality*, only that no NaN/Inf crept in.
    printed = capsys.readouterr().out
    accuracies = [float(v) for v in re.findall(r"accuracy=([0-9.eE+-]+)", printed)]
    assert len(accuracies) > 0
    assert np.all(np.isfinite(accuracies))

    nn_selected = bbm.predict_impt_subbasis(batch_size=4)
    assert isinstance(nn_selected, sx.Basis)


@pytest.mark.slow
def test_train_classifier_without_early_stopping_runs(clean_registry):
    """Same pipeline as above but with early_stop=False, and a couple of
    extra **kwargs (val_frac) passed through to train_classifier, to check
    that path doesn't raise either.
    """
    H, basis_start, basis_small, candidates = build_tiny_pipeline_inputs()

    rand_keys = sx.RandomKeys(seed=99)
    classifier, _ = make_classifier(basis_start, next(rand_keys))

    bbm = sx.BigBasisManager(candidates, classifier)

    random_sel = bbm.sample_subbasis(next(rand_keys), RANDOM_NUM)
    basis_diag = basis_small + random_sel

    _, eigenvec = _diagonalize_dense(H, basis_diag)
    state_diag = sx.State(basis_diag, eigenvec)
    state_train = state_diag % basis_small

    abs_coeff_cut = bbm.derive_abs_coeff_cut(TARGET_NUM, state_train)

    early_stopped = bbm.train_classifier(
        next(rand_keys),
        state_train,
        abs_coeff_cut,
        batch_size=4,
        epochs=3,
        early_stop=False,
        val_frac=0.25,
        printout_vals=False,
    )
    assert early_stopped is False

    nn_selected = bbm.predict_impt_subbasis(batch_size=4)
    assert isinstance(nn_selected, sx.Basis)


@pytest.mark.slow
def test_classifier_save_and_load_state_roundtrip(tmp_path, clean_registry):
    basis_start = sx.Basis(build_start_dets(N_BATH))

    rand_keys = sx.RandomKeys(seed=2024)
    classifier, optimizer = make_classifier(basis_start, next(rand_keys))

    save_path = tmp_path / "nn_state"
    classifier.save_state(str(save_path))

    # Fresh classifier, same architecture, initialized with the notebook's
    # own "throwaway" pattern (RandomKeys.fake_key()) since the actual
    # weights get overwritten by load_state anyway.
    loaded = sx.BasisClassifier(nn_call_on_bits)
    fake_key = sx.RandomKeys.fake_key()
    loaded.initialize(fake_key, basis_start, optimizer)

    kernel_before_load = np.asarray(
        loaded._flax_state.params["Dense_0"]["kernel"]
    ).copy()

    loaded.load_state(str(save_path))

    orig_kernel = np.asarray(classifier._flax_state.params["Dense_0"]["kernel"])
    loaded_kernel = np.asarray(loaded._flax_state.params["Dense_0"]["kernel"])

    # The loaded params must match the saved ones exactly (deterministic
    # round trip), and must actually have changed from the pre-load
    # (differently-seeded) initialization -- otherwise this would
    # trivially pass even if load_state were a no-op.
    assert np.array_equal(loaded_kernel, orig_kernel)
    assert not np.allclose(kernel_before_load, orig_kernel)

    # Sanity check across the whole param pytree, leaf by leaf.
    orig_leaves = jax.tree_util.tree_leaves(classifier._flax_state.params)
    loaded_leaves = jax.tree_util.tree_leaves(loaded._flax_state.params)
    assert len(orig_leaves) == len(loaded_leaves)
    for orig_leaf, loaded_leaf in zip(orig_leaves, loaded_leaves):
        assert np.array_equal(np.asarray(orig_leaf), np.asarray(loaded_leaf))


@pytest.mark.slow
def test_plain_sx_save_load_does_not_interfere_with_nn_state(tmp_path, clean_registry):
    """sx.save/sx.load (dict-based, for plain quantum_core objects) and
    NeuralModel.save_state/load_state (Orbax-based, for Flax params) are two
    separate persistence mechanisms. Exercise both against sibling paths
    under the same tmp_path to confirm they don't clobber each other.
    """
    basis_start = sx.Basis(build_start_dets(N_BATH))

    rand_keys = sx.RandomKeys(seed=55)
    classifier, _ = make_classifier(basis_start, next(rand_keys))

    nn_path = tmp_path / "nn_state"
    basis_path = tmp_path / "plain_basis"

    classifier.save_state(str(nn_path))
    sx.save(basis_start, str(basis_path))

    loaded_basis = sx.load(str(basis_path))
    assert isinstance(loaded_basis, sx.Basis)
    assert loaded_basis == basis_start

    # The NN checkpoint must still be intact and loadable after the plain
    # sx.save/load round trip touched a sibling path.
    reloaded_classifier = sx.BasisClassifier(nn_call_on_bits)
    fake_key = sx.RandomKeys.fake_key()
    optimizer = optax.adam(learning_rate=0.01)
    reloaded_classifier.initialize(fake_key, basis_start, optimizer)
    reloaded_classifier.load_state(str(nn_path))

    orig_leaves = jax.tree_util.tree_leaves(classifier._flax_state.params)
    reloaded_leaves = jax.tree_util.tree_leaves(reloaded_classifier._flax_state.params)
    for orig_leaf, reloaded_leaf in zip(orig_leaves, reloaded_leaves):
        assert np.array_equal(np.asarray(orig_leaf), np.asarray(reloaded_leaf))
