import pandas as _pd
_pd.options.mode.copy_on_write = True

from .analysis import generate_compilation_statistics
from .data_loaders import load_profiler_data
from .protobuf import xla_module_metadata
from .protobuf_utils import compile_protos
from .utils import remove_child_ranges
from .visualization import create_flamegraph, display_flamegraph
