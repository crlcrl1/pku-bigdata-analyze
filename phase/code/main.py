from typing import Callable

import jax
import matplotlib.pyplot as plt
from PIL import Image
import jax.numpy as jnp


def object_value(z: jax.Array, transform: Callable[[jax.Array], jax.Array], y: jax.Array) -> jax.Array:
    m = y.size
    error = jnp.abs(transform(z)) ** 2 - y
    obj_val = jnp.sum(jnp.abs(error))
    return obj_val / m


def get_initial_guess(
        transform: Callable[[jax.Array], jax.Array],
        transform_adj: Callable[[jax.Array], jax.Array],
        y: jax.Array,
        num_iter: int,
        n: int
) -> jax.Array:
    z = jax.random.normal(jax.random.key(0), (n,), dtype=jnp.complex128)
    z = z / jnp.linalg.norm(z)
    for _ in range(num_iter):
        z = transform_adj(y * transform(z))
        z = z / jnp.linalg.norm(z)
    return jnp.sqrt(jnp.sum(y) / y.size) * z


def abs_least_squares(
        transform: Callable[[jax.Array], jax.Array],
        transform_adj: Callable[[jax.Array], jax.Array],
        y: jax.Array,
        n: int,
        step_size: float,
        step_size_schedule: Callable[[int], float],
        num_iter=1000
) -> tuple[jax.Array, list[jax.Array]]:
    m = y.size
    z = get_initial_guess(transform, transform_adj, y, 100, n)
    obj_values = []
    best_value = jnp.inf
    best_z = z

    def grad_fn(z, transform, transform_adj):
        s = transform(z)
        g = jnp.abs(s) ** 2 - y
        grad = transform_adj(jnp.sign(g) * 2 * s)
        grad /= m

        return grad

    for k in range(num_iter):
        current_obj_val = jax.jit(object_value, static_argnames='transform')(z, transform, y)
        obj_values.append(current_obj_val)
        if current_obj_val < best_value:
            best_value = current_obj_val
            best_z = z

        grad = jax.jit(grad_fn, static_argnames=('transform', 'transform_adj'))(
            z, transform, transform_adj)
        step = step_size_schedule(k) * step_size
        z = z - step * grad

    return best_z, obj_values


def rel_error(x: jax.Array, z: jax.Array) -> float:
    phase = jnp.exp(-1j * jnp.angle(jnp.vdot(x, z)))
    return (jnp.linalg.norm(x - phase * z) / jnp.linalg.norm(x)).item()


def plot(obj_values: list[jax.Array]):
    plt.clf()
    plt.plot(obj_values)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Objective Value vs. Iteration Count')
    plt.yscale('log')
    plt.show()


def gaussian():
    key = jax.random.key(0)

    n = 128
    key, subkey = jax.random.split(key)
    key, subkey2 = jax.random.split(key)
    x = jax.random.normal(subkey, (n,), dtype=jnp.float64) + 1j * \
        jax.random.normal(subkey2, (n,), dtype=jnp.float64)

    m = round(4.5 * n)
    key, subkey = jax.random.split(key)
    key, subkey2 = jax.random.split(key)
    A = 1 / jnp.sqrt(2) * (jax.random.normal(subkey, (m, n), dtype=jnp.float64) +
                           1j * jax.random.normal(subkey2, (m, n), dtype=jnp.float64))
    y = jnp.abs(A @ x) ** 2

    @jax.jit
    def transform(z: jax.Array):
        return A @ z

    @jax.jit
    def transform_adj(z: jax.Array):
        return A.conj().T @ z

    def step_size_schedule(k):
        if k <= 2500:
            return 1 / (k + 1)
        else:
            return 1 / (k + 1) ** 1.1

    res, obj_values = abs_least_squares(transform, transform_adj, y, n, 1, step_size_schedule, 5000)
    print(f"error: {rel_error(x, res):.4e}")
    print(f"objective value: {object_value(res, transform, y):.4e}")
    plot(obj_values)


def cdp():
    key = jax.random.key(0)

    n = 128
    x = jax.random.normal(key, (n,), dtype=jnp.float64) + 1j * \
        jax.random.normal(key, (n,), dtype=jnp.float64)

    l = 6
    key, subkey = jax.random.split(key)
    alphabet = jnp.array([1, -1, 1j, -1j])
    masks_indices = jax.random.choice(subkey, len(alphabet), shape=(n, l))
    masks = alphabet[masks_indices]

    key, subkey = jax.random.split(key)
    temp = jax.random.uniform(subkey, (n, l))
    masks = masks * ((temp <= 0.2) * jnp.sqrt(3) + (temp > 0.2) / jnp.sqrt(2))

    @jax.jit
    def transform(z):
        return jnp.fft.fft(masks.conj() * z[:, None], axis=0)

    @jax.jit
    def transform_adj(z):
        return jnp.mean(masks * jnp.fft.ifft(z, axis=0), axis=1)

    def step_size_schedule(k):
        if k <= 1500:
            return 1
        else:
            return 1 / (k - 1500) ** 0.3

    y = jnp.abs(transform(x)) ** 2

    res, obj_values = abs_least_squares(transform, transform_adj, y, n, 1, step_size_schedule, 5000)
    print(f"error: {rel_error(x, res):.4e}")
    print(f"objective value: {object_value(res, transform, y):.4e}")
    plot(obj_values)


def real_image():
    key = jax.random.key(0)

    image = Image.open("ngc6543a.jpg").convert("L")

    image = jnp.array(image, dtype=jnp.float64)
    image /= 255.0

    x_true = image.flatten()
    n = x_true.size
    original_shape = image.shape

    l = 30
    alphabet = jnp.array([1, -1, 1j, -1j])
    key, subkey = jax.random.split(key)
    masks_indices = jax.random.choice(subkey, len(alphabet), shape=(n, l))
    masks = alphabet[masks_indices]

    key, subkey = jax.random.split(key)
    temp = jax.random.uniform(subkey, (n, l))
    masks = masks * ((temp <= 0.2) * jnp.sqrt(3) + (temp > 0.2) / jnp.sqrt(2))

    @jax.jit
    def transform(z_vec: jax.Array):
        return jnp.fft.fft(masks.conj() * z_vec[:, None], axis=0)

    @jax.jit
    def transform_adj(z_mat: jax.Array):
        return jnp.mean(masks * jnp.fft.ifft(z_mat, axis=0), axis=1)

    def step_size_schedule(k):
        if k <= 600:
            return 1
        else:
            return 1 / (k - 600) ** 0.3

    y = jnp.abs(transform(x_true)) ** 2

    num_iterations = 1000
    step_size = 10000.0

    res, obj_values = abs_least_squares(
        transform, transform_adj, y, n, step_size, step_size_schedule, num_iterations)

    print(f"Relative error: {rel_error(x_true, res):.4e}")
    print(f"Final objective value: {object_value(res, transform, y):.4e}")

    plot(obj_values)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.title('Original Image')
    plt.axis('off')

    reconstructed_image = jnp.abs(res).reshape(original_shape)
    reconstructed_image = jnp.clip(reconstructed_image, 0, jnp.max(reconstructed_image))
    if jnp.max(reconstructed_image) > 0:
        reconstructed_image /= jnp.max(reconstructed_image)

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap='gray', vmin=0, vmax=1)
    plt.title(f'Reconstructed Image ({num_iterations} iter)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    jax.config.update("jax_enable_x64", True)
    gaussian()
    cdp()
    real_image()
