import time
from typing import Callable

import numpy as np
from PIL import Image
from numpy.typing import NDArray


def read_image(src_path: str, dest_path: str) -> tuple[NDArray, NDArray]:
    source_img = Image.open(src_path).convert('L')
    source = np.array(source_img)
    dest_img = Image.open(dest_path).convert('L')
    dest = np.array(dest_img)
    return source, dest


def gen_data(src_path: str, dest_path: str) -> tuple[NDArray, NDArray, NDArray]:
    source, dest = read_image(src_path, dest_path)
    rows, cols = source.shape
    source = source.flatten().astype(np.float64)
    dest = dest.flatten().astype(np.float64)
    source += 1e-6
    dest += 1e-6
    source /= np.sum(source)
    dest /= np.sum(dest)

    ii = np.tile(np.arange(rows), cols)
    jj = np.repeat(np.arange(cols), rows)
    cost = (ii[:, None] - ii[None, :]) ** 2 + (jj[:, None] - jj[None, :]) ** 2

    return source, dest, cost


def run_algorithm(src_path: str, dest_path: str, func: Callable, **kwargs):
    source, dest, cost = gen_data(src_path, dest_path)
    alpha = np.array(source)
    beta = np.array(dest)

    start = time.time()
    solution, total_cost, iter_num = func(cost, alpha, beta, **kwargs)
    end = time.time()
    elapsed_time = end - start

    print(f"Total cost: {total_cost:.6f}, Iterations: {iter_num}, Time: {elapsed_time:.4f} seconds")
