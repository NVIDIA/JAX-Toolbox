import sys
import re
import os


def parse_statisics():
    """ Function to parse the statistics results """
    log_file = sys.argv[1]
    total_tests = 0
    errors = 0
    failed_tests = 0
    passed_tests = 0

    with open(log_file, "r") as f:
        content = f.read()
        errors = len(re.findall(r'ERROR:', content))
        failed_tests = len(re.findall(r'FAILED in', content))
        passed_tests = len(re.findall(r'PASSED in', content))
        total_tests = passed_tests + failed_tests

    output = {
        "TOTAL_TESTS": total_tests,
        "ERRORS": errors,
        "PASSED_TESTS": passed_tests,
        "FAILED_TESTS": failed_tests
    }

    with open(os.environ['GITHUB_OUTPUT'], "a") as f:
        for key, value in output.items():
            f.write(f"{key}={value}\n")


if __name__ == "__main__":
    parse_statisics()
