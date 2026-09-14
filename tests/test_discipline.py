"""
The stateless Discipline: evaluation, caching, batching and jacobians.
"""
import numpy as np
import pytest

from msense.api import Discipline, Variable

from disciplines import Parabola, OutputDependent


#
# Statelessness
#

def test_no_per_evaluation_state_on_the_instance():
    disc = Parabola()
    a = disc.eval({"x1": np.array([4.0]), "x2": np.array([5.0])})
    b = disc.eval({"x1": np.array([1.0]), "x2": np.array([1.0])})

    # The first result must not have been disturbed by the second evaluation
    assert a["y"][0] == 41.0 and b["y"][0] == 2.0
    assert not hasattr(disc, "_values")
    assert not hasattr(disc, "_jac")
    assert not hasattr(disc, "_approximating_jac")


def test_eval_does_not_mutate_caller_inputs():
    disc = Parabola()
    inputs = {"x1": np.array([4.0]), "x2": np.array([5.0])}
    before = {k: v.copy() for k, v in inputs.items()}

    disc.eval(inputs)

    for k in inputs:
        assert np.array_equal(inputs[k], before[k])


def test_eval_does_not_mutate_default_inputs():
    disc = Parabola()
    disc.add_default_inputs({"x1": np.array([4.0]), "x2": np.array([5.0])})

    disc.eval({"x1": np.array([9.0]), "x2": np.array([9.0])})

    assert disc.get_default_inputs()["x1"][0] == 4.0


def test_defaults_fill_in_missing_inputs():
    disc = Parabola()
    disc.add_default_inputs({"x1": np.array([4.0]), "x2": np.array([5.0])})

    assert disc.eval({"x1": np.array([1.0])})["y"][0] == 1.0 + 25.0


def test_missing_input_raises():
    disc = Parabola()
    with pytest.raises(KeyError):
        disc.eval({"x1": np.array([1.0])})


def test_wrong_sized_input_raises():
    disc = Parabola()
    with pytest.raises(ValueError):
        disc.eval({"x1": np.array([1.0, 2.0]), "x2": np.array([1.0])})


#
# Cache
#

def test_cache_avoids_re_evaluation():
    disc = Parabola()
    disc.eval({"x1": np.array([4.0]), "x2": np.array([5.0])})
    disc.eval({"x1": np.array([4.0]), "x2": np.array([5.0])})
    assert disc.n_eval == 1


def test_cache_can_be_bypassed():
    disc = Parabola()
    disc.eval({"x1": np.array([4.0]), "x2": np.array([5.0])})
    disc.eval({"x1": np.array([4.0]), "x2": np.array([5.0])}, use_cache=False)
    assert disc.n_eval == 2


def test_cache_hit_and_miss_return_the_same_keys():
    disc = Parabola()
    fresh = disc.eval({"x1": np.array([4.0]), "x2": np.array([5.0])})
    cached = disc.eval({"x1": np.array([4.0]), "x2": np.array([5.0])})
    assert set(fresh) == set(cached) == {"y", "g"}


#
# Batch evaluation
#

def test_eval_batch_uses_the_cache():
    disc = Parabola()
    rows = [{"x1": np.array([float(i)]), "x2": np.array([1.0])} for i in (4, 5, 4)]

    out = disc.eval_batch(rows)

    assert [o["y"][0] for o in out] == [17.0, 26.0, 17.0]
    assert disc.n_eval == 2      # the repeat came from the cache


def test_eval_batch_preserves_order():
    disc = Parabola()
    rows = [{"x1": np.array([float(i)]), "x2": np.array([0.0])}
            for i in range(6, 0, -1)]

    out = disc.eval_batch(rows)

    assert [o["y"][0] for o in out] == [36.0, 25.0, 16.0, 9.0, 4.0, 1.0]


#
# Jacobians
#

@pytest.mark.parametrize("method", [Discipline.DiffMethod.FINITE_DIFFERENCE,
                                    Discipline.DiffMethod.CENTRAL_FINITE_DIFFERENCE,
                                    Discipline.DiffMethod.COMPLEX_STEP])
def test_jacobian_approximation_matches_analytic(method):
    inputs = {"x1": np.array([4.0]), "x2": np.array([5.0])}
    exact = Parabola().differentiate(inputs)

    disc = Parabola()
    disc.set_jacobian_approximation(method, eps=1e-7)
    approx = disc.differentiate(inputs)

    for out_name in ("y", "g"):
        for in_name in ("x1", "x2"):
            assert np.allclose(approx[out_name][in_name],
                               exact[out_name][in_name],
                               rtol=1e-5, atol=1e-6), f"d{out_name}/d{in_name}"


def test_jacobian_approximation_of_a_nonlinear_function():
    """
    d/dx [ exp(x) / sqrt(sin^3 x + cos^3 x) ], checked against the analytic form.
    """
    x = Variable("x")
    f = Variable("f")

    class Function(Discipline):
        def __init__(self):
            super().__init__("Function", [x], [f], cache_type=None)

        def _eval(self, inputs):
            x_ = inputs["x"]
            return {"f": np.exp(x_) / np.sqrt(np.sin(x_)**3 + np.cos(x_)**3)}

    x0 = 1.5
    s, c = np.sin(x0), np.cos(x0)
    denom = s**3 + c**3
    exact = np.exp(x0) * (1.0 - 1.5 * (s**2 * c - c**2 * s) / denom) / np.sqrt(denom)

    for method, eps, rtol in (("finite_difference", 1e-7, 1e-5),
                              ("central_finite_difference", 1e-5, 1e-8),
                              ("complex_step", 1e-20, 1e-12)):
        disc = Function()
        disc.set_jacobian_approximation(method, eps=eps)
        approx = disc.differentiate({"x": np.array([x0])})["f"]["x"][0, 0]
        assert np.isclose(approx, exact, rtol=rtol), f"{method}: {approx} != {exact}"


def test_complex_step_does_not_poison_the_cache():
    disc = Parabola()
    disc.set_jacobian_approximation("complex_step", eps=1e-20)
    disc.differentiate({"x1": np.array([4.0]), "x2": np.array([5.0])})

    outputs = disc.eval({"x1": np.array([4.0]), "x2": np.array([5.0])})
    assert np.isrealobj(outputs["y"]) and outputs["y"][0] == 41.0


#
# Evaluation before differentiation
#

def test_differentiate_supplies_outputs_at_the_same_point():
    """
    A partial expressed through this discipline's own outputs must be correct
    without the caller having evaluated first.
    """
    disc = OutputDependent()
    jac = disc.differentiate({"x1": np.array([3.0])})

    # z = (x1^2)^2 = x1^4  ->  dz/dx1 = 4 x1^3 = 108
    assert np.isclose(jac["z"]["x1"][0, 0], 108.0)


def test_differentiate_reuses_a_cached_evaluation():
    disc = OutputDependent()
    inputs = {"x1": np.array([3.0])}

    disc.eval(inputs)
    assert disc.n_eval == 1

    disc.differentiate(inputs)
    assert disc.n_eval == 1, "the pre-differentiation evaluation should hit the cache"


def test_analytic_jacobian_is_cached():
    disc = Parabola()
    inputs = {"x1": np.array([4.0]), "x2": np.array([5.0])}

    disc.differentiate(inputs)
    disc.differentiate(inputs)

    assert disc.n_diff == 1
