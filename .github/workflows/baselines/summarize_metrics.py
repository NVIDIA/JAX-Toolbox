import os
import json
import glob
import sys
from statistics import mean
from test_utils import read_tb_tag, read_e2e_time


def _create_summary(loss, train_time, e2e_time):
    steps = list(loss.keys())
    intervals = [k2 - k1 for k1, k2 in zip(loss.keys(), steps[1:])]
    assert all(i == intervals[0] for i in intervals)

    baseline = {
        "start_step": steps[0],
        "end_step": steps[-1],
        "step_interval": intervals[0],
        "loss_values": list(loss.values()),
        "step_times": list(train_time.values()),
        "step_time_avg": mean(list(train_time.values())),
        "e2e_time_seconds": e2e_time,
    }
    return baseline


def main():
    loss_summary_name = "loss"
    train_time_summary_name = "Steps/sec"
    if sys.argv[1]:
        test_config = sys.argv[1]
    else:
        sys.exit(1)

    try:
        event_file = os.path.join(test_config, "summaries/train/events*")
        event_file = glob.glob(event_file)[0]
        print(f'EVENT FILE: {event_file}')
        loss = read_tb_tag(event_file, loss_summary_name)
        train_time = read_tb_tag(event_file, train_time_summary_name)
        e2e_time = read_e2e_time(test_config + ".log")

        baseline = _create_summary(loss, train_time, e2e_time)
        json_fname = test_config + "_metrics.json"
        print(f'JSON FILENAME: {json_fname}')

        with open(json_fname, "w") as f:
            json.dump(baseline, f)

    except KeyError as e:
        print(e)
        print("Run might have failed, see", test_config)


if __name__ == "__main__":
    main()
