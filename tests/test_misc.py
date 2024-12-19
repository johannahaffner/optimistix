import jax
import jax.numpy as jnp
import optimistix._misc as optx_misc

from .helpers import tree_allclose


def test_inexact_asarray_no_copy():
    x = jnp.array([1.0])
    assert optx_misc.inexact_asarray(x) is x
    y = jnp.array([1.0, 2.0])
    assert jax.vmap(optx_misc.inexact_asarray)(y) is y


# See JAX issue #15676
def test_inexact_asarray_jvp():
    p, t = jax.jvp(optx_misc.inexact_asarray, (1.0,), (2.0,))
    assert type(p) is not float
    assert type(t) is not float


def test_tree_clip():
    # Test with a scalar
    lower = 0.0
    upper = 5.0
    x = 10.0
    y = optx_misc.tree_clip(x, lower, upper)
    assert y == upper

    # Test with arrays
    lower = jnp.array([0.0, 0.0])
    upper = jnp.array([5.0, 5.0])
    x = jnp.array([10.0, 10.0])
    y = optx_misc.tree_clip(x, lower, upper)
    assert jnp.all(y == upper)

    # Test with a PyTree
    lower = {"a": 0.0, "b": 0.0}
    upper = {"a": 5.0, "b": 5.0}
    x = {"a": 10.0, "b": 10.0}
    y = optx_misc.tree_clip(x, lower, upper)
    assert y == upper

    # Test with a complex PyTree
    lower = {
        "a": jnp.array(-10.0),
        "b": {"c": jnp.array([3.0, 4.0]), "d": jnp.array(-5.0)},
    }
    upper = {
        "a": jnp.array(10.0),
        "b": {"c": jnp.array([5.0, 6.0]), "d": jnp.array(5.0)},
    }
    x = {"a": -20.0, "b": {"c": -jnp.array([10.0, 20.0]), "d": -10.0}}
    y = optx_misc.tree_clip(x, lower, upper)
    assert tree_allclose(y, lower)
