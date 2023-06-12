# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools as ft
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
import numpy as np
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Float, Int, PyTree

from .._custom_types import Aux, Fn, Out, Y
from .._least_squares import AbstractLeastSquaresSolver
from .._line_search import AbstractDescent, AbstractLineSearch, OneDimensionalFunction
from .._minimise import AbstractMinimiser, minimise
from .._misc import (
    max_norm,
    sum_squares,
    tree_full_like,
    tree_where,
    tree_zeros,
)
from .._solution import RESULTS
from .descent import UnnormalisedNewton
from .iterative_dual import DirectIterativeDual, IndirectIterativeDual
from .learning_rate import LearningRate
from .misc import compute_jac_residual
from .trust_region import ClassicalTrustRegion


def _is_struct(x):
    return eqx.is_array(x) or isinstance(x, jax.ShapeDtypeStruct)


class _GNState(eqx.Module):
    descent_state: PyTree
    vector: PyTree[ArrayLike]
    operator: lx.AbstractLinearOperator
    diff: PyTree[Array]
    result: RESULTS
    f_val: PyTree[Array]
    f_prev: PyTree[Array]
    next_init: Array
    aux: Any
    step: Int[Array, ""]


class AbstractGaussNewton(AbstractLeastSquaresSolver[_GNState, Y, Out, Aux]):
    rtol: float
    atol: float
    line_search: AbstractLineSearch
    descent: AbstractDescent
    norm: Callable

    def init(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> _GNState:
        del options
        f0 = jnp.array(jnp.inf)
        aux = tree_zeros(aux_struct)
        # Dummy vector and operator for first pass. Note that having `jnp.ones_like`
        # is preferable to `jnp.zeros_like` as the latter can lead to linear solves
        # of the form `0 x = 0` which can return `nan` values.
        vector = jtu.tree_map(lambda x: jnp.ones(x.shape, x.dtype), f_struct)
        operator_struct = eqx.filter_eval_shape(
            lx.JacobianLinearOperator, fn, y, args, _has_aux=True
        )
        dynamic_operator_struct, static_operator = eqx.partition(
            operator_struct, _is_struct
        )
        dynamic_operator = jtu.tree_map(
            lambda x: jnp.ones(x.shape, x.dtype), dynamic_operator_struct
        )
        operator = eqx.combine(dynamic_operator, static_operator)
        descent_state = self.descent.init_state(fn, y, vector, operator, None, args, {})
        return _GNState(
            descent_state=descent_state,
            vector=vector,
            operator=operator,
            diff=tree_full_like(y, jnp.inf),
            result=RESULTS.successful,
            f_val=f0,
            f_prev=f0,
            next_init=jnp.array(1.0),
            aux=aux,
            step=jnp.array(0),
        )

    def step(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _GNState,
        tags: frozenset[object],
    ) -> tuple[Y, _GNState, Aux]:
        descent = eqx.Partial(
            self.descent,
            descent_state=state.descent_state,
            args=args,
            options=options,
        )

        def line_search_fn(x, args):
            residual, aux = fn(x, args)
            return sum_squares(residual), aux

        problem_1d = OneDimensionalFunction(line_search_fn, descent, y)
        line_search_options = {
            "f0": jnp.where(state.step > 1, state.f_val, jnp.inf),
            "compute_f0": (state.step == 1),
            "vector": state.vector,
            "operator": state.operator,
            "diff": y,
        }
        line_search_options["predicted_reduction"] = ft.partial(
            self.descent.predicted_reduction,
            descent_state=state.descent_state,
            args=args,
            options={},
        )
        init = jnp.where(
            state.step <= 1,
            self.line_search.first_init(state.vector, state.operator, options),
            state.next_init,
        )
        line_sol = minimise(
            fn=problem_1d,
            has_aux=True,
            solver=self.line_search,
            y0=init,
            args=args,
            options=line_search_options,
            max_steps=100,
            throw=False,
        )
        # `new_aux` and `f_val` are the output of of at `f` at step the
        # end of the line search. ie. they are not `f(y)`, but rather
        # `f(y_new)` where `y_new` is `y` in the next call of `step`. In
        # other words, we use FSAL.
        (f_val, diff, new_aux, _, next_init) = line_sol.aux
        new_y = tree_where(state.step > 0, (ω(y) + ω(diff)).ω, y)
        vector, operator, _ = compute_jac_residual(fn, new_y, args)
        descent_state = self.descent.update_state(
            state.descent_state,
            state.diff,
            vector,
            operator,
            operator_inv=None,
            options=options,
        )
        new_state = _GNState(
            descent_state=descent_state,
            vector=vector,
            operator=operator,
            diff=diff,
            result=RESULTS.successful,
            f_val=f_val,
            f_prev=state.f_val,
            next_init=next_init,
            aux=new_aux,
            step=state.step + 1,
        )
        # Notice that this is `state.aux`, not `new_state.aux` or `aux`.
        # we delay the return of `aux` by one step because of the FSAL
        # in the line search.
        # We want aux at `f(y)`, but line_search returns
        # `aux` at `f(y_new)`
        return new_y, new_state, state.aux

    def terminate(
        self,
        fn: Fn[Y, Out, Aux],
        y: Y,
        args: PyTree,
        options: dict[str, Any],
        state: _GNState,
        tags: frozenset[object],
    ):
        at_least_two = state.step >= 2
        y_scale = (self.atol + self.rtol * ω(y).call(jnp.abs)).ω
        y_converged = self.norm((state.diff**ω / y_scale**ω).ω) < 1
        f_scale = self.rtol * jnp.abs(state.f_prev) + self.atol
        f_converged = (jnp.abs(state.f_val - state.f_prev) / f_scale) < 1
        converged = y_converged & f_converged
        linsolve_fail = state.result != RESULTS.successful
        terminate = linsolve_fail | (converged & at_least_two)
        result = RESULTS.where(linsolve_fail, state.result, RESULTS.successful)
        return terminate, result

    def buffers(self, state: _GNState) -> tuple[()]:
        return ()


class GaussNewton(AbstractGaussNewton):
    rtol: float
    atol: float
    line_search: AbstractMinimiser = LearningRate(np.array(1.0))
    descent: AbstractDescent = UnnormalisedNewton(gauss_newton=True)
    norm: Callable = max_norm


class LevenbergMarquardt(AbstractGaussNewton):
    def __init__(
        self,
        rtol: float,
        atol: float,
        norm=max_norm,
        backtrack_slope: float = 0.1,
        decrease_factor: float = 0.5,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = DirectIterativeDual(gauss_newton=True)
        self.line_search = ClassicalTrustRegion()


class IndirectLevenbergMarquardt(AbstractGaussNewton):
    line_search: AbstractMinimiser
    descent: AbstractDescent

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm=max_norm,
        lambda_0: Float[ArrayLike, ""] = 1e-3,
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.line_search = ClassicalTrustRegion()
        self.descent = IndirectIterativeDual(
            gauss_newton=True,
            lambda_0=lambda_0,
        )