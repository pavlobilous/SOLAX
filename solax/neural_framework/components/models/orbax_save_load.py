import os
import orbax.checkpoint as ocp


def save_flax_state(fld, state):
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
    abs_path = os.path.abspath(fld)
    orbax_path = ocp.test_utils.epath.gpath.PosixGPath(abs_path) / "1"
    
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    restored_state = ckptr.restore(
        orbax_path, args=ocp.args.StandardRestore(state)
    )
    return restored_state