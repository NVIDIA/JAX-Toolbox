import os
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'test_config', help='Directory corresponding to artifacts of a single test configuration, e.g. 1DP1TP1PP')
    parser.add_argument('-l', '--loss_summary_name', default='loss',
                              help='Key in the tensorboard event file containing loss data')
    parser.add_argument('-p', '--perf_summary_name', default='Steps/sec',
                        help='Key in the tensorboard event file containing perf data')
    args = parser.parse_args()

    try:
        searchpath = os.path.join(args.test_config, "train")
        if not os.path.exists(searchpath):
            searchpath = os.path.join(args.test_config, "summaries/train")
        assert os.path.exists(searchpath), f"Neither {args.test_config}/train nor {args.test_config}/summaries/train dirs exist"
        event_files = glob.glob(os.path.join(searchpath, "events*"))
        assert len(event_files) > 0, f"{searchpath} did not contain a tensorboard events file"

        event_file = event_files[0]
        print(f'EVENT FILE: {event_file}')
        loss = read_tb_tag(event_file, args.loss_summary_name)
        train_time = read_tb_tag(event_file, args.perf_summary_name)
        e2e_time = read_e2e_time(args.test_config + ".log")

        baseline = _create_summary(loss, train_time, e2e_time)
        json_fname = args.test_config + "_metrics.json"
        print(f'JSON FILENAME: {json_fname}')
        with open(json_fname, "w") as f:
            json.dump(baseline, f)

    except KeyError as e:
        print(e)
        print("Run might have failed, see", args.test_config)


if __name__ == "__main__":
    main()
