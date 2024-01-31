import os
import sys
import numpy as np
import json
import argparse

def main(config, run_dirs, output_dir):
    # Store metrics data as list of dicts
    json_fnames = [f"{r}/{config}_metrics.json" for r in run_dirs]
    src_data = []
    for fname in json_fnames:
        with open(fname, "r") as f:
            src_data.append(json.load(f))

    # Ensure start step, end step, interval equal across runs
    src_data
    for k in ["start_step", "end_step", "step_interval"]:
        values = [metrics[k] for metrics in src_data]
        print("checking equality for", k)
        print(values)
        assert all([v == values[0] for v in values])

    # Gather metrics across dirs
    avg_data = src_data[0].copy()  # Use first metrics dict as a template
    loss_data = np.array([metrics["loss_values"] for metrics in src_data])
    step_times_data = np.array([metrics["step_times"] for metrics in src_data])
    mean_step_times_data = np.array([metrics["step_time_avg"] for metrics in src_data])
    e2e_time_data = np.array([metrics["e2e_time_seconds"] for metrics in src_data])

    # Average
    avg_data["loss_values"] = list(np.mean(loss_data, axis=0))
    avg_data["step_times"] = list(np.mean(step_times_data, axis=0))
    avg_data["step_time_avg"] = np.mean(mean_step_times_data)
    avg_data["e2e_time_seconds"] = np.mean(e2e_time_data)

    # save to file
    os.makedirs(output_dir, exist_ok=True)
    output_baseline_path = os.path.join(output_dir, config + '.json')
    with open(output_baseline_path, "w") as f:
        json.dump(avg_data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config name like 1N1G. This would correspond to $workflow_id/$artifact_name/$config')
    parser.add_argument('--run_dirs', type=str, required=True, nargs='+', help='One or more workflow run dirs of the form $workflow_id/$artifact_name')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to place the averaged baseline. E.g., for upstream-pax it would be PAX_MGMN/upstream (relative to this dir)')
    args = parser.parse_args()
    main(args.config, args.run_dirs, args.output_dir)
