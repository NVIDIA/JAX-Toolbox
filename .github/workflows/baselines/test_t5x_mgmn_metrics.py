import pytest
import os
import json
import glob
import test_utils
from statistics import mean
from numpy.testing import assert_allclose

LOSS_RTOL = {
    '1G1N': 0.10,  # orig = 0.02
    '1G2N': 0.10,  # orig = 0.03
    '1P1G': 0.10,  # orig = 0.03
    '1P2G': 0.10,  # orig = 0.03
    '1P4G': 0.10,  # orig = 0.035
    '1P8G': 0.10,  # orig = 0.035
    '2G1N': 0.10,  # orig = 0.025
    '2G2N': 0.10,  # orig = 0.015
    '4G1N': 0.10,  # orig = 0.03
    '4G2N': 0.10,  # orig = 0.03
    '8G1N': 0.10,  # orig = 0.03
    '8G2N': 0.10,  # orig = 0.05
}
STEP_TIME_MULT = {
    "1G1N": 0.95,
    "1G2N": 0.95,
    "1P1G": 0.95,
    "1P2G": 0.95,
    "1P4G": 0.95,
    "1P8G": 0.95,
    "2G1N": 0.95,
    "2G2N": 0.95,
    "4G1N": 0.95,
    "4G2N": 0.95,
    "8G1N": 0.95,
    "8G2N": 0.95,
}
E2E_TIME_MULT = {
    "1G1N": 0.95,
    "1G2N": 0.95,
    "1P1G": 0.95,
    "1P2G": 0.95,
    "1P4G": 0.95,
    "1P8G": 0.95,
    "2G1N": 0.95,
    "2G2N": 0.95,
    "4G1N": 0.95,
    "4G2N": 0.95,
    "8G1N": 0.95,
    "8G2N": 0.95,
}
test_dir = os.path.dirname(os.path.abspath(__file__))
baselines_dir = os.path.join(test_dir, os.environ.get("BASELINES_DIR"))
results_dir = os.environ.get("RESULTS_DIR")
loss_summary_name = "loss"
step_time_summary_name = "timing/steps_per_second"


@pytest.mark.parametrize("baseline_filename", os.listdir(baselines_dir))
def test_loss(baseline_filename):
    baseline_filepath = os.path.join(baselines_dir, baseline_filename)
    test_config = baseline_filename.split(".")[0]
    event_file = os.path.join(results_dir, test_config, "train/events*")
    event_file = glob.glob(event_file)[0]
    with open(baseline_filepath, "r") as baseline_file:
        baseline_data = json.load(baseline_file)
        loss_expected_values = baseline_data["loss_values"]
        start_step = baseline_data["start_step"]
        end_step = baseline_data["end_step"]
        interval = baseline_data["step_interval"]
        loss_expected = {step: loss_expected_values[i] for i, step in enumerate(
            range(start_step, end_step+1, interval))}
        loss_actual = test_utils.read_tb_tag(event_file, loss_summary_name)

        assert loss_expected.keys() == loss_actual.keys(), \
            f"Steps at which loss was emitted for run do not match baseline. \
            Actual steps: {loss_actual.keys()}, Baseline steps: {loss_expected.keys()}"
        assert_allclose(list(loss_actual.values()), list(loss_expected.values()),
                        rtol=LOSS_RTOL[test_config],
                        err_msg=f"Run loss values: {loss_actual.values()}, \
                                Baseline loss values: {loss_expected.values()}")


@pytest.mark.parametrize("baseline_filename", os.listdir(baselines_dir))
def test_step_time(baseline_filename):
    baseline_filepath = os.path.join(baselines_dir, baseline_filename)
    test_config = baseline_filename.split(".")[0]
    event_file = os.path.join(results_dir, test_config, "train/events*")
    event_file = glob.glob(event_file)[0]
    with open(baseline_filepath, "r") as baseline_file:
        step_time_avg_expected = json.load(baseline_file)["step_time_avg"]
        step_time_values = test_utils.read_tb_tag(event_file, step_time_summary_name).values()
        step_time_avg_actual = mean(step_time_values)
        assert step_time_avg_actual > step_time_avg_expected * \
            STEP_TIME_MULT[test_config], f"Step time values: {step_time_values} (Avg: {step_time_avg_actual}), Expected avg: {step_time_avg_expected}"


@pytest.mark.parametrize("baseline_filename", os.listdir(baselines_dir))
def test_e2e_time(baseline_filename):
    baseline_filepath = os.path.join(baselines_dir, baseline_filename)
    test_config = baseline_filename.split(".")[0]
    run_log = os.path.join(results_dir, test_config + ".log")
    with open(baseline_filepath, "r") as baseline_file:
        e2e_time_expected = json.load(baseline_file)["e2e_time_seconds"]
        e2e_time_actual = test_utils.read_e2e_time(run_log)
        assert e2e_time_actual < e2e_time_expected / \
            E2E_TIME_MULT[test_config], f"Run E2E time: {e2e_time_actual}, Expected E2E time: {e2e_time_expected}"
