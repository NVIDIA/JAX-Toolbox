import argparse
import glob
import json
import os


def count_tests_from_exit_status_files(exit_status_files) -> tuple:
    """
    Counts the number of passed and failed tests from exit status files.

    Args:
        exit_status_files (list): List of file patterns containing exit status.

    Returns:
        tuple: (passed_tests, failed_tests, total_tests)
    """
    passed_tests = 0
    failed_tests = 0
    total_tests = len(exit_status_files)
    for status_file in exit_status_files:
        with open(status_file) as f:
            status_data = json.load(f)
            state = status_data.get('state')
            exitcode = status_data.get('exitcode')
            if state == 'COMPLETED' and exitcode == 0:
                passed_tests += 1
            else:
                failed_tests += 1
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


def determine_badge_color(passed_tests: int, failed_tests: int, total_tests: int):
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
        badge_color, badge_message
    """
    if failed_tests > 0 or total_tests == 0:
        badge_message = 'error'
        badge_color = 'red'
    else:
        badge_message = f"{passed_tests}/{total_tests} passed"
        if failed_tests == 0:
            badge_color = 'brightgreen'
        else:
            badge_color = 'yellow'

    return badge_color, badge_message


def write_exit_status_summary(exit_status_files, summary_filename, fw_name, github_step_summary=None):
    """
    Generates a markdown summary of the exit status files. This function works for multi-node sitreps

    Args:
        exit_status_files (list): List of exit status json files.
        summary_filename (str): The filename to write the summary to. TODO shoudl we keep this constant?
        fw_name (str): Framework name to include in the summary.
        github_step_summary (str): Path to GITHUB_STEP_SUMMARY file (if any).
    """
    with open(summary_filename, 'w') as f:
        f.write(f"\n\n## {fw_name} MGMN+SPMD Test Status\n")
        f.write("| Test Case | State | Exit Code |\n")
        f.write("| --- | --- | --- |\n")

        for status_file in exit_status_files:
            # Files are named <FW_NAME>-<GHID>-<NAME>/<NAME>-status.json
            test_case = os.path.basename(status_file).replace('-status.json', '')
            with open(status_file, 'r') as sf:
                data = json.load(sf)
                state = data.get('state')
                exitcode = data.get('exitcode')
            f.write(f"| {test_case} | {state} | {exitcode} |\n")

    # TODO append to GITHUB_STEP_SUMMARY
    if github_step_summary and os.path.exists(github_step_summary):
        with open(github_step_summary, 'a') as f_out:
            with open(summary_filename, 'r') as f_in:
                f_out.write(f_in.read())


def write_metrics_summary(metrics_files: list, 
                          summary_md_filename: str, 
                          summary_json_filename: str, 
                          fw_name:str, 
                          github_step_summary=None):
    """
    Generates a markdown and json summary of metrics files.

    Args:
        metrics_files (list): List of metrics json files.
        # TODO should we keep these two constant?
        summary_md_filename (str): The filename to write the markdown summary to. This is "metrics_summary.md" 
        summary_json_filename (str): The filename to write the json summary to. This is "metrics_summary.json"
        fw_name (str): Framework name to include in the summary.
        github_step_summary (str): Path to GITHUB_STEP_SUMMARY file (if any).
    """
    all_metrics = []
    header = None
    # TODO improve readability of this ufnction
    with open(summary_md_filename, 'w') as f_md:
        f_md.write(f"## {fw_name} MGMN Test Metrics\n")
        print_row = lambda lst: f_md.write('| ' + ' | '.join(str(el) for el in lst) + ' |\n')

        for path in metrics_files:
            with open(path) as f:
                obj = json.load(f)
                all_metrics.append(obj)
                if not header:
                    header = list(obj.keys())
                    print_row(["Job Name"] + header)
                    print_row(["---"] * (1 + len(header)))
                job_name = os.path.basename(path)[:-len('_metrics.json')]
                print_row([job_name] + [obj[h] for h in header])

        f_md.write('NOTE: Average step time includes compilation time and thus may be an underestimate of true performance\n')

    # Write the json summary
    with open(summary_json_filename, 'w') as f_json:
        json.dump(all_metrics, f_json, indent=4)

    # Optionally append to GITHUB_STEP_SUMMARY
    if github_step_summary and os.path.exists(github_step_summary):
        with open(github_step_summary, 'a') as f_out:
            with open(summary_md_filename, 'r') as f_in:
                f_out.write(f_in.read())


def main() -> None:
    """
    Main entry point 
    """
    parser = argparse.ArgumentParser(description='Generate sitrep and badge JSON files.')
    parser.add_argument('--badge_label', required=True, help='Label for the badge')
    parser.add_argument('--badge_filename', required=True, help='Output badge filename')
    parser.add_argument('--sitrep_filename', default='sitrep.json', help='Output sitrep filename')
    parser.add_argument('--exit_status_patterns', nargs='*', default=['**/*-status.json'], help='Tests with error output')
    parser.add_argument('--metrics_logs', nargs='*', default=['metrics-*/*.log'], help='Metrics log file(s)')
    parser.add_argument('--badge_message', help='Badge message (overrides default)')
    parser.add_argument('--badge_color', help='Badge color (overrides default)')
    parser.add_argument('--exit_status_summary_file', default="exit_status_summary.json", help='Output exit status summary markdown file')
    parser.add_argument('--metrics_summary_file', help='Output metrics summary markdown file')
    parser.add_argument('--tags', help='Tags from the build')
    parser.add_argument('--digest', help='Digest from the build')
    parser.add_argument('--outcome', help='Outcome of the build')
    # mgmn parameters
    parser.add_argument('--github_run_id', default=os.environ.get('GITHUB_RUN_ID'), help='GitHub Run ID')
    parser.add_argument('--github_output_file', default=os.environ.get('GITHUB_OUTPUT'), help='GitHub output file for actions')
    parser.add_argument('--github_step_summary', default=os.environ.get('GITHUB_STEP_SUMMARY'), help='GitHub step summary file')
    # optional parameters 
    parser.add_argument('--total_tests', default=None, help='Total number of tests')
    parser.add_argument('--errors', default=None, help='Number of errors')
    parser.add_argument('--failed_tests', default=None, help='Number of failed tests')
    parser.add_argument('--passed_tests', default=None, help='Number of passed tests')


    args = parser.parse_args()

    # Set default patterns if not provided
    if not args.exit_status_patterns:
        args.exit_status_patterns = [f"{args.badge_label}*-{args.github_run_id}-*/*-status.json"]
    if not args.metrics_logs:
        args.metrics_logs = [f"{args.badge_label}-metrics-test-log/report.jsonl"]
    if not args.metrics_json_patterns:
        args.metrics_json_patterns = [f"{args.badge_label}-metrics-test-log/*_metrics.json"]

    # if we have outcome, then we can produce the badge immediately 
    if args.outcome: 
        sitrep_data = {
            'summary': f"{args.badge_label}: pass" if args.outcome == "success" else f"{args.badge_label}: fail",
            'badge_label': args.badge_label,
            'tags': args.tags,
            'digest': args.digest,
            'outcome': args.outcome,
        }
        with open(args.sitrep_filename, 'w') as f:
            json.dump(sitrep_data, f, indent=2)

        badge_data = {
            'schemaVersion': 1,
            'label': args.badge_label,
            'message': "pass" if args.outcome == "success" else "fail",
            'color': "brightgreen" if args.outcome == "success" else "red"
        }
        with open(args.badge_filename, 'w') as f:
            json.dump(badge_data, f, indent=2)
        return

    # Count exit status tests
    exit_status_files = []
    for pattern in args.exit_status_patterns:
        exit_status_files.extend(glob.glob(pattern, recursive=True))

    # Count metrics tests
    metrics_logs = []
    for pattern in args.metrics_logs:
        metrics_logs.extend(glob.glob(pattern, recursive=True))

    # Collect metrics JSON files
    metrics_files = []
    for pattern in args.metrics_json_patterns:
        metrics_files.extend(glob.glob(pattern, recursive=True))

    # Write exit status summary
    if args.exit_status_summary_file:
        write_exit_status_summary(exit_status_files, args.exit_status_summary_file, args.badge_label, args.github_step_summary)
    
    # Write metrics summary
    if args.metrics_summary_file:
        write_metrics_summary(metrics_files, args.metrics_summary_file, 'metrics_summary.json', args.badge_label, args.github_step_summary)

    # Count the number of tests passed to determine the success
    if not args.passed_tests and not args.failed_tests and not args.total_tests:
        passed_tests, failed_tests, total_tests = count_tests_from_exit_status_files(exit_status_files)

        badge_color, badge_message = determine_badge_color(
            passed_tests, failed_tests, total_tests
        )
    else: 
        passed_tests = args.passed_tests
        failed_tests = args.failed_tests
        total_tests = args.total_tests
        badge_color, badge_message = determine_badge_color(
            passed_tests, failed_tests, total_tests
        )

    summary = f"{args.badge_label}: {badge_message}"

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
        'tags': args.tags,
        'digest': args.digest,
        'outcome': args.outcome,
    }
    if args.errors: 
        sitrep_data['errors'] = args.errors

    with open(args.sitrep_filename, 'w') as f:
        json.dump(sitrep_data, f, indent=2)

    badge_data = {
        'schemaVersion': 1,
        'label': args.badge_label,
        'message': badge_message,
        'color': args.badge_color or badge_color
    }

    with open(args.badge_filename, 'w') as f:
        json.dump(badge_data, f, indent=2)

    # github_output = os.environ.get('GITHUB_OUTPUT')
    # if github_output:
    #     with open(github_output, 'a') as fh:
    #         print(f'STATUS={status}', file=fh)

    # Check and display metrics summary
    if os.path.exists('metrics_summary.json'):
        print("metrics_summary.json exists:")
        with open('metrics_summary.json', 'r') as f:
            print(f.read())
    else:
        print("metrics_summary.json does not exist.")


if __name__ == "__main__":
    main()
