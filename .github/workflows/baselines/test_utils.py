from tensorboard.backend.event_processing import event_accumulator
from tensorboard.util import tensor_util
import numpy as np

# By default TB tries to be smart about what to load in memory to avoid OOM
# Since we expect every step to be there when we do our comparisons, we explicitly
# set the size guidance to 0 so that we load everything. It's okay given our tests
# are small/short.
size_guidance = {
    event_accumulator.TENSORS: 0,
    event_accumulator.SCALARS: 0,
}


def read_tb_tag(tb_file: str, summary_name: str) -> dict:
    ea = event_accumulator.EventAccumulator(tb_file, size_guidance)
    ea.Reload()

    return {
        event.step: tensor_util.make_ndarray(event.tensor_proto).item()
        for event in ea.Tensors(summary_name)
    }

def read_maxtext_tb_tag(tb_file: str, summary_name: str) -> dict:
    ea = event_accumulator.EventAccumulator(tb_file, size_guidance)
    ea.Reload()

    return {
        event.step: np.asarray(event.value).item()
        for event in ea.Scalars(summary_name)
    }

def read_e2e_time(log_file: str) -> float:
    with open(log_file, "r") as log:
        for line in reversed(list(log)):
            if line.startswith("real"):
                minutes = line.split()[1].split('m')[0]
                seconds = line.split('m')[1].split('s')[0]
                return float(minutes) * 60 + float(seconds)
    return -100000000
