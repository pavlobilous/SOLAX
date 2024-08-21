from jax import jit, vmap, pmap

from .ladder_mappings import *
from .commut_phases import *


map_with_ladseq_vOpt = jax.vmap(
    map_with_ladseq,
    in_axes=(None, 0, None)
)

map_with_ladseq_vDet_vOpt = jax.vmap(
    map_with_ladseq_vOpt,
    in_axes=(0, None, None)
)

map_with_ladseq_pvDet_vOpt = jax.pmap(
    map_with_ladseq_vDet_vOpt,
    in_axes=(0, None, None)
)


ladseq_phase_vOpt=jax.vmap(
    ladseq_phase,
    in_axes=(None, None, 0)
)

ladseq_phase_vDet_vOpt =jax.vmap(
    ladseq_phase_vOpt,
    in_axes=(0, None, None)
)

ladseq_phase_pvDet_vOpt =jax.pmap(
    ladseq_phase_vDet_vOpt,
    in_axes=(0, None, None),
    static_broadcasted_argnums=(1,)
)