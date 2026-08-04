import numpy as np


def pad_to_fixed_size(array, target_shape, pad_value=0):
    """
    Pad an array to a fixed shape with a specified pad value.

    Parameters
    ----------
    array : array-like
        Array to pad
    target_shape : tuple
        Desired output shape
    pad_value : int or float, optional
        Value to use for padding, by default 0

    Returns
    -------
    padded_array : array-like
        Padded array with desired shape
    """
    pad_width = [(0, max(0, t - s)) for s, t in zip(array.shape, target_shape)]
    return np.pad(array, pad_width, constant_values=pad_value)


def process_in_chunks(array, chunk_size):
    """
    Yield chunks of the array with a fixed size, padding the last chunk if necessary.

    Parameters
    ----------
    array : array-like
        Array to process in chunks
    chunk_size : int
        Size of each chunk

    Yields
    ------
    chunk : array-like
        Array chunk of fixed size (padded if necessary)
    """
    n = array.shape[0]
    for i in range(0, n, chunk_size):
        chunk = array[i : i + chunk_size]
        if chunk.shape[0] < chunk_size:
            chunk = pad_to_fixed_size(chunk, (chunk_size,) + chunk.shape[1:])
        yield chunk
