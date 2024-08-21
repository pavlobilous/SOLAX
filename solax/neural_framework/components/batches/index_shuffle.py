from dataclasses import dataclass, KW_ONLY

import numpy as np
import jax
import jax.numpy as jnp


def chunk_params(length, max_ind):
    chunks_num = -(length // -max_ind)
    last_chunk_sz = length - (chunks_num - 1) * max_ind
    return chunks_num, last_chunk_sz


def shuffled_chunks(key, chunks_num, last_chunk_sz):
    sh_chunks = []
    for chunk in range(chunks_num):
        chunk_sz = max_ind \
                     if chunk < chunks_num - 1 \
                     else last_chunk_sz
        key, subkey = jax.random.split(key)
        sh_chunks.append(
            np.asarray(
                jax.random.permutation(subkey, chunk_sz)
            ) 
        )
    return sh_chunks


def vertical_shuffle(key, arr):
    inds = jax.random.permutation(key,
            jnp.tile(jnp.arange(arr.shape[0]), (arr.shape[1], 1)).T,
            axis=0,
            independent=True
        )
    return np.take_along_axis(arr, inds, axis=0)


def shuffled_inds(key, *, length: int, max_ind: int = None):
    
    if not length:
        raise ValueError("Nothing to shuffle.")
    
    if not max_ind:
        max_ind = jnp.iinfo(jnp.array(123).dtype).max
    
    chunks_num, last_chunk_sz = chunk_params(length, max_ind)
    
    key, subkey = jax.random.split(key)
    sh_chunks = shuffled_chunks(subkey, chunks_num, last_chunk_sz)
    
    if chunks_num > 1:
        sh_chunks = [ch + i * max_ind for i, ch in enumerate(sh_chunks)]
        
        arr_left = np.vstack([arr[:last_chunk_sz] for arr in sh_chunks])
        key, subkey = jax.random.split(key)
        arr_left = vertical_shuffle(subkey, arr_left)

        arr_right = np.vstack([arr[last_chunk_sz:] for arr in sh_chunks[:-1]])
        key, subkey = jax.random.split(key)
        arr_right = vertical_shuffle(subkey, arr_right)
        
        return np.concatenate([arr_left.reshape(-1), arr_right.reshape(-1)])
    
    else:
        return sh_chunks[0]