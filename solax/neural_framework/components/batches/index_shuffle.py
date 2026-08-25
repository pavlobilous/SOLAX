"""
Random shuffling of index sequences, including sequences longer than
jax.random.permutation can handle directly: shuffled_inds() splits a
"length"-long range into chunks of at most "max_ind" indices, shuffles
each chunk independently (jax.random.permutation is used per chunk),
and then reassembles/re-shuffles across chunk boundaries so the result
is a uniformly shuffled permutation of range(length) as a whole.
"""
import numpy as np
import jax
import jax.numpy as jnp


def chunk_params(length, max_ind):
    """
    Splits a range of size "length" into chunks of at most "max_ind"
    indices each. Returns (chunks_num, last_chunk_sz): the number of
    chunks and the size of the last (possibly shorter) chunk; all
    other chunks have size "max_ind".
    """
    chunks_num = -(length // -max_ind)
    last_chunk_sz = length - (chunks_num - 1) * max_ind
    return chunks_num, last_chunk_sz


def shuffled_chunks(key, chunks_num, max_ind, last_chunk_sz):
    """
    Builds "chunks_num" independently-shuffled index chunks (each a
    permutation of range(chunk size), as a NumPy array), splitting
    "key" via jax.random.split for each chunk. All chunks except the
    last have full size "max_ind" and the last one has size
    "last_chunk_sz".
    """
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
    """
    Independently shuffles each column of the 2D array "arr" along
    axis 0 (rows), using "key". Used by shuffled_inds() to re-shuffle
    index chunks across chunk boundaries once they've been stacked
    into a 2D array.
    """
    inds = jax.random.permutation(key,
            jnp.tile(jnp.arange(arr.shape[0]), (arr.shape[1], 1)).T,
            axis=0,
            independent=True
        )
    return np.take_along_axis(arr, inds, axis=0)


def shuffled_inds(key, *, length: int, max_ind: int = None):
    """
    Returns a uniformly-shuffled permutation of range("length") as a
    NumPy 1D array of indices, using "key" for randomness.

    Input:

        - "key": a jax.random key.
        - "length": size of the index range to shuffle. Raises
            ValueError if 0/None (falsy).
        - "max_ind" (default=None): largest chunk size
            jax.random.permutation is asked to shuffle at once (see
            chunk_params()); if 0/None (falsy), defaults to the max
            value of jnp's default int dtype (about 2^31), which for
            any realistic "length" keeps everything in a single chunk.

    Output:
        A NumPy 1D array of "length" indices, a permutation of
        range("length").
    """
    if not length:
        raise ValueError("Nothing to shuffle.")
    
    if not max_ind:
        max_ind = jnp.iinfo(jnp.array(123).dtype).max
    
    chunks_num, last_chunk_sz = chunk_params(length, max_ind)
    
    key, subkey = jax.random.split(key)
    sh_chunks = shuffled_chunks(subkey, chunks_num, max_ind, last_chunk_sz)
    
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