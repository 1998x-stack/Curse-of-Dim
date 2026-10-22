import json
import os
import subprocess
import sys

import pytest

from curse_of_dim.experiments import EXPERIMENTS, run_selected


def test_experiment_registry():
    assert list(EXPERIMENTS) == [
        "geometry", "distances", "knn_noise", "sample_complexity",
        "intrinsic_dimension", "double_descent", "conditioning",
        "nonlinear_manifold", "embedding_anisotropy", "effective_dimension",
        "autoencoder_manifold",
    ]


def test_reject_unknown_before_running(tmp_path):
    with pytest.raises(ValueError, match="Unknown experiment"):
        run_selected(["missing"], out=tmp_path)


def test_geometry_run_generates_figure_and_metrics(tmp_path):
    metrics = run_selected(["geometry"], out=tmp_path)
    assert (tmp_path / "01_geometry.png").stat().st_size > 5_000
    assert 0 < metrics["geometry"]["r_med_d2"] < metrics["geometry"]["r_med_d100"] < 0.5


def test_distance_concentration_is_observable(tmp_path):
    metrics = run_selected(["distances"], out=tmp_path)["distances"]
    assert metrics["cv_d512"] < metrics["cv_d2"]
    assert metrics["near_far_d512"] > metrics["near_far_d2"]


def test_cli_list_and_json(tmp_path):
    env = {**os.environ, "MPLBACKEND": "Agg", "OPENBLAS_NUM_THREADS": "2"}
    listed = subprocess.run([sys.executable, "-m", "curse_of_dim", "--list"],
                            check=True, capture_output=True, text=True, env=env)
    assert "intrinsic_dimension" in listed.stdout
    out = tmp_path / "cli"
    subprocess.run([sys.executable, "-m", "curse_of_dim", "-e", "geometry", "--out", str(out)],
                   check=True, capture_output=True, text=True, env=env)
    assert "geometry" in json.loads((out / "metrics.json").read_text())
