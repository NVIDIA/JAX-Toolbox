import pytest
import os
import json
import glob
import sys
import test_utils
from statistics import mean

STEP_TIME_MULT = 0.95
E2E_TIME_MULT = 0.95
test_dir = os.path.dirname(os.path.abspath(__file__))
baselines_dir = os.path.join(test_dir, os.environ.get("BASELINES_DIR"))
results_dir = os.environ.get("RESULTS_DIR")
loss_summary_name = "loss"
step_time_summary_name = "Steps/sec"


@pytest.mark.parametrize("baseline_filename", os.listdir(baselines_dir))
def test_loss(baseline_filename):
    baseline_filepath = os.path.join(baselines_dir, baseline_filename)
    test_config = baseline_filename.split(".")[0]
    event_file = os.path.join(results_dir, test_config, "summaries/train/events*")
    event_file = glob.glob(event_file)[0]
    with open(baseline_filepath, "r") as baseline_file:
        end_step = json.load(baseline_file)["end_step"]
        loss_actual = test_utils.read_tb_tag(event_file, loss_summary_name)
        assert 0 <= loss_actual[end_step] < 1.8e-6, f"Loss at final step: {loss_actual[end_step]}, Expected 0 <= loss < 1.8e-6"


@pytest.mark.parametrize("baseline_filename", os.listdir(baselines_dir))
def test_step_time(baseline_filename):
    baseline_filepath = os.path.join(baselines_dir, baseline_filename)
    test_config = baseline_filename.split(".")[0]
    event_file = os.path.join(results_dir, test_config, "summaries/train/events*")
    event_file = glob.glob(event_file)[0]
    with open(baseline_filepath, "r") as baseline_file:
        step_time_avg_expected = json.load(baseline_file)["step_time_avg"]
        step_time_values = test_utils.read_tb_tag(event_file, step_time_summary_name).values()
        step_time_avg_actual = mean(step_time_values)
        assert step_time_avg_actual > step_time_avg_expected * \
            STEP_TIME_MULT, f"Step time values: {step_time_values} (Avg: {step_time_avg_actual}), Expected avg: {step_time_avg_expected}"


'''@pytest.mark.parametrize("baseline_filename", os.listdir(baselines_dir))
def test_e2e_time(baseline_filename):
    baseline_filepath = os.path.join(baselines_dir, baseline_filename)
    test_config = baseline_filename.split(".")[0]
    run_log = os.path.join(results_dir, test_config + ".log")
    with open(baseline_filepath, "r") as baseline_file:
        e2e_time_expected = json.load(baseline_file)["e2e_time_seconds"]
        e2e_time_actual = test_utils.read_e2e_time(run_log)
        assert e2e_time_actual < e2e_time_expected / \
            E2E_TIME_MULT, f"Run E2E time: {e2e_time_actual}, Expected E2E time: {e2e_time_expected}"'''
