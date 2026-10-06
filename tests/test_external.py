"""
ExternalDiscipline: run directories, failure detection, batch dispatch.

These tests drive a real subprocess: a small Python script standing in for a
solver executable. It reads a JSON input file and writes a JSON result file, so
the mechanics under test are the real ones.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from msense.api import CachePolicy, EvaluationFailure, Variable
from msense.ext.external import ExternalDiscipline

x1 = Variable("x1", lb=0.0, ub=10.0)
x2 = Variable("x2", lb=0.0, ub=10.0)
y = Variable("y")


SOLVER = r'''
import json, sys, time
with open("inputs.json") as f:
    d = json.load(f)

x1, x2 = d["x1"], d["x2"]

if d.get("hang"):
    time.sleep(30)

if x1 > 5.0:                      # "diverged"
    sys.stderr.write("residual did not converge\n")
    sys.exit(1)

if x2 > 5.0:                      # exits cleanly but writes nothing
    sys.exit(0)

with open("results.json", "w") as f:
    json.dump({"y": x1 ** 2 + x2 ** 2}, f)
'''


class Solver(ExternalDiscipline):
    """
    Wraps the stand-in solver script above.
    """

    def __init__(self, script: Path, work_dir: Path, hang: bool = False, **kwargs):
        self.hang = hang
        kwargs.setdefault("cache_policy", CachePolicy.FULL)
        super().__init__("Solver", [x1, x2], [y],
                         command=[sys.executable, str(script)],
                         work_dir=str(work_dir),
                         expected_outputs=["results.json"],
                         cache_path=str(work_dir / "cache.json"),
                         **kwargs)

    def write_inputs(self, inputs, run_dir):
        (run_dir / "inputs.json").write_text(json.dumps({
            "x1": float(inputs["x1"][0]),
            "x2": float(inputs["x2"][0]),
            "hang": self.hang}))

    def read_outputs(self, inputs, run_dir):
        d = json.loads((run_dir / "results.json").read_text())
        return {"y": np.array([d["y"]])}


@pytest.fixture
def script(tmp_path):
    path = tmp_path / "solver.py"
    path.write_text(SOLVER)
    return path


def at(x1_val, x2_val):
    return {"x1": np.array([float(x1_val)]), "x2": np.array([float(x2_val)])}


#
# A successful run
#

def test_evaluates_through_the_executable(script, tmp_path):
    disc = Solver(script, tmp_path / "work")
    assert disc.eval(at(2, 3))["y"][0] == 13.0


def test_each_run_gets_its_own_directory(script, tmp_path):
    work = tmp_path / "work"
    disc = Solver(script, work)
    disc.eval(at(2, 3))
    disc.eval(at(1, 1))

    run_dirs = sorted(p.name for p in work.iterdir() if p.is_dir())
    assert len(run_dirs) == 2 and run_dirs[0] != run_dirs[1]
    # Inputs are isolated, not overwritten in a shared directory
    written = {json.loads((work / d / "inputs.json").read_text())["x1"]
               for d in run_dirs}
    assert written == {2.0, 1.0}


def test_a_log_is_written_for_every_run(script, tmp_path):
    work = tmp_path / "work"
    disc = Solver(script, work)
    disc.eval(at(2, 3))

    logs = list(work.glob("*/run.log"))
    assert len(logs) == 1 and "returncode] 0" in logs[0].read_text()


def test_successful_runs_can_be_cleaned_up(script, tmp_path):
    work = tmp_path / "work"
    disc = Solver(script, work, cleanup_successful_runs=True)
    disc.eval(at(2, 3))

    assert not [p for p in work.iterdir() if p.is_dir()]


#
# Failure detection
#

def test_a_nonzero_return_code_is_a_failure(script, tmp_path):
    disc = Solver(script, tmp_path / "work")
    with pytest.raises(EvaluationFailure, match="returned 1"):
        disc.eval(at(9, 1))
    assert disc.n_fail == 1


def test_a_missing_output_file_is_a_failure_not_a_parser_error(script, tmp_path):
    """
    The solver exits 0 but writes nothing. Without the expected_outputs check
    this surfaces as a FileNotFoundError from read_outputs.
    """
    disc = Solver(script, tmp_path / "work")
    with pytest.raises(EvaluationFailure, match="results.json"):
        disc.eval(at(1, 9))


def test_a_timeout_is_a_failure(script, tmp_path):
    disc = Solver(script, tmp_path / "work", hang=True, timeout=1.0)
    with pytest.raises(EvaluationFailure, match="timeout"):
        disc.eval(at(1, 1))


def test_a_failed_run_is_kept_for_triage(script, tmp_path):
    work = tmp_path / "work"
    disc = Solver(script, work, cleanup_successful_runs=True)
    with pytest.raises(EvaluationFailure):
        disc.eval(at(9, 1))

    logs = list(work.glob("*/run.log"))
    assert len(logs) == 1
    assert "residual did not converge" in logs[0].read_text()


def test_a_failure_can_be_tolerated(script, tmp_path):
    disc = Solver(script, tmp_path / "work")
    outputs = disc.eval(at(9, 1), tolerate_failure=True)
    assert np.isnan(outputs["y"][0])


def test_a_failure_is_cached(script, tmp_path):
    disc = Solver(script, tmp_path / "work")
    disc.eval(at(9, 1), tolerate_failure=True)
    disc.eval(at(9, 1), tolerate_failure=True)
    assert disc.n_eval == 1, "a known-bad design should not be re-run"


def test_a_missing_executable_is_a_configuration_error(tmp_path):
    disc = Solver(Path("/nonexistent/solver.py"), tmp_path / "work")
    disc.command = ["/nonexistent/executable"]
    # Not an EvaluationFailure: the design is fine, the setup is not
    with pytest.raises(OSError):
        disc.eval(at(1, 1))


#
# Batch evaluation
#

def test_batch_preserves_order(script, tmp_path):
    disc = Solver(script, tmp_path / "work")
    out = disc.eval_batch([at(1, 1), at(2, 2), at(3, 3)])
    assert [o["y"][0] for o in out] == [2.0, 8.0, 18.0]


def test_batch_collapses_repeats_onto_one_run(script, tmp_path):
    disc = Solver(script, tmp_path / "work")
    out = disc.eval_batch([at(2, 2), at(3, 3), at(2, 2), at(2, 2)])

    assert [o["y"][0] for o in out] == [8.0, 18.0, 8.0, 8.0]
    assert disc.n_eval == 2, "repeated design vectors should run once"


def test_batch_uses_the_cache(script, tmp_path):
    disc = Solver(script, tmp_path / "work")
    disc.eval(at(2, 2))
    disc.eval_batch([at(2, 2), at(3, 3)])
    assert disc.n_eval == 2


def test_batch_runs_concurrently(script, tmp_path):
    work = tmp_path / "work"
    disc = Solver(script, work, max_workers=4)
    rows = [at(i, 0) for i in range(5)]

    out = disc.eval_batch(rows)

    assert [o["y"][0] for o in out] == [0.0, 1.0, 4.0, 9.0, 16.0]
    # Five isolated run directories, none clobbered
    assert len([p for p in work.iterdir() if p.is_dir()]) == 5


def test_batch_tolerates_failures_among_successes(script, tmp_path):
    disc = Solver(script, tmp_path / "work", max_workers=4)
    rows = [at(1, 1), at(9, 1), at(2, 2), at(8, 1)]

    out = disc.eval_batch(rows, tolerate_failure=True)

    assert out[0]["y"][0] == 2.0 and out[2]["y"][0] == 8.0
    assert np.isnan(out[1]["y"][0]) and np.isnan(out[3]["y"][0])
    assert disc.n_fail == 2


def test_batch_failure_propagates_when_not_tolerated(script, tmp_path):
    disc = Solver(script, tmp_path / "work", max_workers=2)
    with pytest.raises(EvaluationFailure):
        disc.eval_batch([at(1, 1), at(9, 1)])


def test_batch_results_are_independent_copies(script, tmp_path):
    disc = Solver(script, tmp_path / "work")
    out = disc.eval_batch([at(2, 2), at(2, 2)])

    out[0]["y"][0] = -1.0
    assert out[1]["y"][0] == 8.0
