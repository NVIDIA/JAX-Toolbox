import subprocess
import tempfile


def nsys_jax(command):
    """
    Helper to run nsys-jax with a unique output file that will be automatically
    cleaned up on destruction.
    """
    output = tempfile.NamedTemporaryFile(suffix=".zip")
    subprocess.run(
        ["nsys-jax", "--force-overwrite", "--output", output.name] + command, check=True
    )
    return output
