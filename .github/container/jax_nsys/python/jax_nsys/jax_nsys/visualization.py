from IPython.display import display, IFrame
import subprocess
from typing import Iterable
import xml.etree.ElementTree

from .protobuf_utils import which


def create_flamegraph(
    data: dict[Iterable[str], float], title: str, filename: str, width: int = 1200
) -> tuple[str, IFrame]:
    """
    Given a data structure of the form
      {
        (A, B): 1.0,
        (A, B, C): 2.0,
        ...
      }
    flatten it into the
      A;B 1.0
      A;B;C 2.0
      ...
    format expected by flamegraph.pl and generate an interactive SVG from it
    with the given width.

    Returns a tuple (svg_data, InlineIFrame), where the latter can be passed to
    IPython.display.display(...) to be rendered inline in a Jupyter notebook.
    """
    flat_data = ""
    for loc, value in data.items():
        assert not any(";" in x for x in loc)
        assert value >= 0.0, f"Negative value {value} under {loc}"
        flat_data += ";".join(map(str, loc)) + " " + str(value) + "\n"
    svg = subprocess.run(
        [which("flamegraph.pl"), f"--width={width}", f"--title={title}"],
        check=True,
        input=flat_data,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    tree = xml.etree.ElementTree.fromstring(svg)
    assert int(tree.attrib["width"]) == width
    height = int(tree.attrib["height"])
    # TODO: should be able to generate <iframe srcdoc="..."> and avoid the
    # temporary file, but my attempt to do that ended up with extra scrollbars
    # on the IFrame
    with open(filename, "w") as ofile:
        ofile.write(svg)
    return svg, IFrame(filename, width, height)


def display_flamegraph(**kwargs):
    svg, iframe = create_flamegraph(**kwargs)
    display(iframe)
