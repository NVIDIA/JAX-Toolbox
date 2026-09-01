import os
import sys
import zipfile

helper_dir = os.path.join(os.path.dirname(__file__), "nsys_jax_test_helpers")
if helper_dir not in sys.path:
    sys.path.insert(0, helper_dir)
from nsys_jax_test_helpers import nsys_jax_with_result

# Deletes its own source file, so that the XLA metadata refers to a file that no longer
# exists when nsys-jax gathers source code.
self_deleting_program = """
import jax
import os


@jax.jit
def self_deleting_function(x):
    return x @ x.T


square = jax.random.normal(jax.random.key(1), (32, 32))
square = self_deleting_function(square)
os.remove(__file__)
"""


def test_missing_source_file_is_not_fatal(capfd, tmp_path):
    program = tmp_path / "self_deleting_program.py"
    program.write_text(self_deleting_program)
    output, result = nsys_jax_with_result(
        [sys.executable, str(program)], out_dir=tmp_path
    )
    assert not program.exists()
    result.check_returncode()
    stdout, _ = capfd.readouterr()
    assert str(program) in stdout
    with zipfile.ZipFile(output.name) as archive:
        assert "sources" + str(program) not in archive.namelist()
