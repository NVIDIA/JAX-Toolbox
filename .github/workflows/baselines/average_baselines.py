import os
import sys
import numpy as np
import json

def main():
    if len(sys.argv) < 3:
        sys.exit(1)

    config = sys.argv[1]
    run_dirs = sys.argv[2:]

    # Store metrics data as list of dicts
    json_fnames = [f"{r}/{config}_metrics.json" for r in run_dirs]
    src_data = []
    for fname in json_fnames:
        with open(fname, "r") as f:
            src_data.append(json.load(f))

    # TODO: Ensure start step, end step, interval equal across runs
    assert ...

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
    fname = config + ".json"
    with open(fname, "w") as f:
        json.dump(avg_data, f)

if __name__ == "__main__":
    main()
