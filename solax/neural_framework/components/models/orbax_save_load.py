"""
Orbax-based checkpointing of a Flax TrainState: saving/loading the
raw pytree of parameters and optimizer state to/from a directory on
disk. This is a separate, independent persistence mechanism from
solax.save()/solax.load() (which handle plain, dict-based quantum_core
objects); the two can safely be used against sibling paths without
interfering with each other.
"""
import os
import orbax.checkpoint as ocp


def save_flax_state(fld, state):
    """
    Saves a Flax TrainState "state" to directory "fld" (an actual
    checkpoint subdirectory "1" is created under it). "fld" is erased
    and recreated empty first (ocp.test_utils.erase_and_create_empty),
    so any pre-existing contents there are discarded. Saving is done
    asynchronously via an AsyncCheckpointer, but this function blocks
    until the save completes (ckptr.wait_until_finished()) before
    returning. Returns None.
    """
    abs_path = os.path.abspath(fld)
    orbax_path = ocp.test_utils.erase_and_create_empty(abs_path) / "1"

    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    ckptr.save(
        orbax_path,
        args=ocp.args.StandardSave(state)
    )
    ckptr.wait_until_finished()
    return None


def load_flax_state(fld, state):
    """
    Loads a Flax TrainState previously saved by save_flax_state() from
    directory "fld" (reading its "1" checkpoint subdirectory). "state"
    is used as the pytree structure/dtype template Orbax restores
    into -- typically a freshly initialize()d NeuralModel's state, not
    a state you intend to keep. Returns the restored state (a new
    TrainState-like object; "state" itself is not mutated).
    """
    abs_path = os.path.abspath(fld)
    orbax_path = ocp.test_utils.epath.gpath.PosixGPath(abs_path) / "1"

    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    restored_state = ckptr.restore(
        orbax_path, args=ocp.args.StandardRestore(state)
    )
    return restored_state