import numpy as np
from numpy.typing import NDArray
from PIL import Image
import matplotlib.pyplot as plt


def linear_time_svd(A: NDArray, c: int, p: NDArray, k: int) -> tuple[NDArray, NDArray]:
    if p.shape[0] != A.shape[1]:
        raise ValueError("p must have the same number of columns as A")

    # sample c columns from A with probability p
    sampled_indices = np.random.choice(A.shape[1], size=c, replace=False, p=p)
    sampled_col = A[:, sampled_indices] / np.sqrt(c * p[sampled_indices])

    eig_vals, eig_vecs = np.linalg.eigh(sampled_col.T @ sampled_col)
    eig_vals = np.sqrt(eig_vals[::-1][:k])
    eig_vecs = sampled_col @ eig_vecs[:, :k] / eig_vals

    return eig_vals, eig_vecs


def random_matrix(c: int, k: int):
    m = 2048
    n = 512
    r = 20
    np.random.seed(0)
    A = np.random.randn(m, r) @ np.random.randn(r, n)
    A_norm = np.linalg.norm(A, ord="fro")
    p = np.linalg.norm(A, axis=0) ** 2 / A_norm ** 2
    eig_vals, eig_vecs = linear_time_svd(A, c, p, k)
    real_eig_vals = np.linalg.svdvals(A)
    real_eig_vals = real_eig_vals[:k]
    relative_error = np.abs(eig_vals - real_eig_vals) / real_eig_vals
    return relative_error


def image_matrix(c: int, k: int):
    img = Image.open("figure.jpg").convert("L")
    img = np.array(img)
    p = np.linalg.norm(img, axis=0) ** 2 / np.linalg.norm(img, ord="fro") ** 2
    eig_vals, eig_vecs = linear_time_svd(img, c, p, k)
    real_eig_vals = np.linalg.svdvals(img)
    real_eig_vals = real_eig_vals[:k]
    relative_error = np.abs(eig_vals - real_eig_vals) / real_eig_vals
    return relative_error


if __name__ == "__main__":
    err1 = random_matrix(50, 20)
    err2 = random_matrix(100, 20)
    err3 = random_matrix(150, 20)
    err4 = random_matrix(200, 20)
    # err1 = image_matrix(50, 20)
    # err2 = image_matrix(100, 20)
    # err3 = image_matrix(150, 20)
    # err4 = image_matrix(200, 20)

    plt.plot(err1, label="c=50", marker='o')
    plt.plot(err2, label="c=100", marker='o')
    plt.plot(err3, label="c=150", marker='o')
    plt.plot(err4, label="c=200", marker='o')
    plt.xlabel("k")
    plt.ylabel("Relative error")
    plt.title("Relative error of eigenvalues")
    plt.legend()
    plt.savefig("relative_error_matrix.pdf")
