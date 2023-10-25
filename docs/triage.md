# Triage Workflow

There is a Github Action Workflow called [_triage.yaml](../.github/workflows/_triage.yaml) that can
be used to help determine if a test failure was due to a change in (t5x or pax) or further-up, e.g., in (Jax or CUDA). This workflow is not the end-all, and further investigation is usually needed, 
but this automates the investigation of questions like "what state of library X works with Jax at state Y?"


## Algorithm
The pseudocode for the triaging algorithm is as follows:
```python
# Broken pax + jax
BROKEN_NIGHTLY = 'ghcr.io/nvidia/pax:nightly-YYYY-MM-05'
# Working pax + jax
WORKING_NIGHTLY = 'ghcr.io/nvidia/pax:nightly-YYYY-MM-01'

for container between(WORKING_NIGHTLY, BROKEN_NIGHTLY):
    new_container = fast_forward_pax_in(container)
    test_result = run_pax_tests_on(new_container)
    if test_result == "Pass":
        return "Suspect: Newer Jax containers"
else:
    return "Suspect: New change in pax"
```

__Note__: Since we are working with mutliple repositories, we cannot use binary-search to search over
the containers because the assumption that the test_results for all containers between the working and broken is monotonic, is not guaranteed. So the only logical choice is to linearly scan thru the
images between `WORKING_NIGHTLY` and `BROKEN_NIGHTLY`.

## How to use it
There are two ways the triage workflow can be used:

1. As a [re-usable workflow](https://docs.github.com/en/actions/using-workflows/reusing-workflows)
(example: [nightly-pax-test-mgmn.yaml](../.github/workflows/nightly-pax-test-mgmn.yaml)). Existing
workflows will trigger the `_triage.yaml` workflow if the tests fail.
2. Or triggered from the web-ui [here](https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/_triage.yaml).

### Inspecting the output
After the job is finished, you can inspect the summary of the run and there should be a table
like [this](https://github.com/NVIDIA/JAX-Toolbox/actions/runs/6152793581#summary-16699715277) for pax
or like [this](https://github.com/NVIDIA/JAX-Toolbox/actions/runs/6152785988#summary-16698903888) for t5x.

Both should show a table like this:
| Rewind to | Test result | Image |
| --- | --- | --- |
| nightly-2023-07-18 | success | ghcr.io/nvidia/jax-toolbox-internal:6087387492-nightly-2023-07-18-ff-t5x-to-2023-07-20 |
| nightly-2023-07-19 | success | ghcr.io/nvidia/jax-toolbox-internal:6087387492-nightly-2023-07-19-ff-t5x-to-2023-07-20 |
| | failure <br> (assumed broken) | ghcr.io/nvidia/t5x:nightly-2023-07-20 (BROKEN_IMAGE) |

Where "Rewind to" is which nightly we started from and then fast-forwarded the libraries to;
"Test result" is the updated test result with this new fast-forwarded image; and "Image" is
the updated image with fast-fowarded code.

The last row in this table always has "Test result" = `failure (assumed broken)`, since we
do not need to re-run tests for the BROKEN_IMAGE since the user would have already observed
the failure.

There are two conclusions you can make from this table:
1. If any row has `Test result=success`, then that older nightly tag under `Rewind to` works with the
state of the libraries in `BROKEN_IMAGE`. The latest dated "success" can be used as a starting point
for a git-bisect to investigate the issue.
2. If all rows have `Test result=failure`, then a change in the libraries (e.g., pax/t5x) likely
caused the failure and should be investigated.