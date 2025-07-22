from .args import parse_args
from .utils import get_logger
from .triage_tool import TriageTool


def main() -> None:
    """
    Main entry point for the triage tool.
    """
    args = parse_args()
    logger = get_logger(args.output_prefix)
    tool = TriageTool(args, logger)
    passing_url, failing_url = tool.find_container_range()
    passing_versions, failing_versions = tool.gather_version_info(
        passing_url, failing_url
    )
    tool.run_version_bisection(passing_versions, failing_versions)
