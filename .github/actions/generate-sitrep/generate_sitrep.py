import argparse
import glob
import json
import os
from pathlib import Path

def count_tests_from_exit_status_files(exit_status_patterns) -> tuple:
    """
    Counts the number of passed and failed tests from exit status files.

    Args:
    exit_status_patterns (list): List of file patterns containing exit status.

    Returns:
    tuple: (passed_tests, failed_tests, total_tests)
    """
    passed_tests = 0
    failed_tests = 0
    for status_file in exit_status_patterns:
        with open(status_file) as f:
            status_data = json.load(f)
            state = status_data.get('state')
            exitcode = status_data.get('exitcode')
            if state == 'COMPLETED' and exitcode == 0:
                passed_tests += 1
            else:
                failed_tests += 1
    total_tests = len(exit_status_patterns)
    return passed_tests, failed_tests, total_tests


def count_tests_from_metrics_logs(metrics_logs) -> tuple:
    """
    Counts the number of passed and failed tests from metrics logs.

    Args:
    metrics_logs (list): List of metrics log files.

    Returns:
    tuple: (pytest_passed_tests, pytest_failed_tests, pytest_total_tests)
    """
    pytest_passed_tests = 0
    pytest_failed_tests = 0
    pytest_total_tests = 0
    for metrics_log in metrics_logs:
        with open(metrics_log) as f:
            for line in f:
                data = json.loads(line)
                if data.get('$report_type') == 'TestReport' and data.get('when') == 'call':
                    outcome = data.get('outcome')
                    if outcome == 'passed':
                        pytest_passed_tests += 1
                    elif outcome == 'failed':
                        pytest_failed_tests += 1
                    pytest_total_tests += 1
    return pytest_passed_tests, pytest_failed_tests, pytest_total_tests


def determine_badge_color(passed_tests: int, failed_tests: int, total_tests: int,
                          pytest_passed_tests: int = None, pytest_failed_tests: int = None, pytest_total_tests: int = None) -> tuple:
    """
    Determines the badge color based on test results.

    Args:
        passed_tests (int): Number of passed tests.
        failed_tests (int): Number of failed tests.
        total_tests (int): Total number of tests.
        pytest_passed_tests (int): Number of passed pytest tests (default=None).
        pytest_failed_tests (int): Number of failed pytest tests (default=None).
        pytest_total_tests (int): Total number of pytest tests (default=None).

    Returns:
        tuple: (badge_color, status)
    """
    if (failed_tests == 0 and total_tests > 0) and \
       (pytest_failed_tests == 0 and pytest_total_tests > 0 if pytest_total_tests else True):
        return 'brightgreen', 'success'
    elif passed_tests == 0 or (pytest_passed_tests == 0 if pytest_passed_tests is not None else False):
        return 'red', 'failure'
    else:
        return 'yellow', 'failure'


def main() -> None:
    """
    Main entry point 
    """
    parser = argparse.ArgumentParser(description='Generate sitrep and badge JSON files.')
    parser.add_argument('--badge_label', required=True, help='Label for the badge')
    parser.add_argument('--exit_status_patterns', help='Tests with error output')
    parser.add_argument('--metrics_log', action='append', help='Metrics log file(s)')
    parser.add_argument('--badge_filename', required=True, help='Output badge filename')
    parser.add_argument('--sitrep_filename', default='sitrep.json', help='Output sitrep filename')
    parser.add_argument('--full_result_markdown', help='File containing full result markdown')
    parser.add_argument('--total_tests', type=int, help='Total number of tests')
    parser.add_argument('--passed_tests', type=int, help='Number of passed tests')
    parser.add_argument('--failed_tests', type=int, help='Number of failed tests')
    parser.add_argument('--badge_message', help='Badge message (overrides default)')
    parser.add_argument('--badge_color', help='Badge color (overrides default)')
    parser.add_argument('--exit_status_summary_file', help='Output exit status summary markdown file')
    parser.add_argument('--metrics_summary_file', help='Output metrics summary markdown file')

    args = parser.parse_args()

    # Count exit status tests
    exit_status_tests = []
    if args.exit_status_patterns:
        for pattern in args.exit_status_patterns.split(','):
            exit_status_tests.extend(glob.glob(pattern))

    # Count metrics tests
    metrics_tests = []
    if args.metrics_log:
        metrics_tests = args.metrics_log

    # Initialize test counts
    passed_tests = 0
    failed_tests = 0
    total_tests = 0
    pytest_passed_tests = 0
    pytest_failed_tests = 0
    pytest_total_tests = 0

    # Count exit status tests
    if exit_status_tests:
        passed_tests, failed_tests, total_tests = count_tests_from_exit_status_files(exit_status_tests)

    # Count metrics tests
    if metrics_tests:
        pytest_passed_tests, pytest_failed_tests, pytest_total_tests = count_tests_from_metrics_logs(metrics_tests)

        # Generate metrics summary markdown
        if args.metrics_summary_file:
            with open(args.metrics_summary_file, 'w') as f:
                f.write(f"## {args.badge_label} MGMN Test Metrics\n")
                # Assuming you have metrics JSON files to summarize
                metrics_files_pattern = f"{args.badge_label}-metrics-test-log/*_metrics.json"
                metrics_files = glob.glob(metrics_files_pattern)
                if metrics_files:
                    all_metrics = []
                    header = None
                    f.write("| Job Name | Metrics |\n")
                    f.write("| --- | --- |\n")
                    for path in metrics_files:
                        with open(path) as mf:
                            obj = json.load(mf)
                            all_metrics.append(obj)
                            job_name = Path(path).stem.replace('_metrics', '')
                            if not header:
                                header = obj.keys()
                                f.write("| Job Name | " + " | ".join(header) + " |\n")
                                f.write("| --- " * (len(header) + 1) + "|\n")
                            f.write(f"| {job_name} | " + " | ".join(str(obj[h]) for h in header) + " |\n")
                else:
                    f.write("No metrics files found.\n")

    # Generate exit status summary markdown
    if args.exit_status_summary_file:
        with open(args.exit_status_summary_file, 'w') as f:
            f.write(f"\n\n## {args.badge_label} MGMN+SPMD Test Status\n")
            f.write("| Test Case | State | Exit Code |\n")
            f.write("| --- | --- | --- |\n")
            for status_file in exit_status_tests:
                test_case = Path(status_file).stem.replace('-status', '')
                with open(status_file) as f_status:
                    status_data = json.load(f_status)
                    state = status_data.get('state')
                    exitcode = status_data.get('exitcode')
                    f.write(f"| {test_case} | {state} | {exitcode} |\n")

    badge_color, status = determine_badge_color(
        passed_tests, failed_tests, total_tests,
        pytest_passed_tests, pytest_failed_tests, pytest_total_tests
    )

    badge_message = f"{passed_tests}/{total_tests} jobs | {pytest_passed_tests}/{pytest_total_tests} metrics"
    summary = f"# {args.badge_label} MGMN Test: {badge_message}"

    full_result_markdown = ''
    if args.exit_status_summary_file and os.path.exists(args.exit_status_summary_file):
        with open(args.exit_status_summary_file, 'r') as f:
            full_result_markdown += f.read()
    if args.metrics_summary_file and os.path.exists(args.metrics_summary_file):
        with open(args.metrics_summary_file, 'r') as f:
            full_result_markdown += f.read()

    sitrep_data = {
        'summary': summary,
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': failed_tests,
        'badge_label': args.badge_label,
        'badge_color': badge_color,
        'badge_message': badge_message,
        'full_result_markdown': full_result_markdown,
    }

    with open(args.sitrep_filename, 'w') as f:
        json.dump(sitrep_data, f, indent=2)

    badge_data = {
        'schemaVersion': 1,
        'label': args.badge_label,
        'message': badge_message,
        'color': badge_color
    }

    with open(args.badge_filename, 'w') as f:
        json.dump(badge_data, f, indent=2)

    github_output = os.environ.get('GITHUB_OUTPUT')
    if github_output:
        with open(github_output, 'a') as fh:
            print(f'STATUS={status}', file=fh)
            print(f'TOTAL_TESTS={total_tests}', file=fh)


if __name__ == "__main__":
    main()
