# Triage tool

`jax-toolbox-triage` is a tool to automate the process of attributing regressions to an
individual commit of JAX or XLA.
It takes as input a command that returns an error (non-zero) code when run in "recent"
containers, but which returns a success (zero) code when run in some "older" container.
The command must be executable within the containers, *i.e.* it cannot refer to files
that only exist on the host system, unless those are explicitly mounted in using the
`-v` (`--container-mount`) option.

The tool follows a three-step process:
  1. A container-level search backwards from the "recent" container where the test is
     known to fail, which identifies an "older" container where the test passes. This
     search proceeds with an exponentially increasing step size and is based on the
     `YYYY-MM-DD` tags under `ghcr.io/nvidia/jax`.
  2. A container-level binary search to refine this to the **latest** available
     container where test passes and the **earliest** available container where it
     fails.
  3. A commit-level binary search, repeatedly building + testing inside the same
     container, to identify a single commit of a software package known to the tool
     (JAX, XLA, Flax, optionally MaxText) that causes the test to start failing, and a
     set of reference commits of the other packages that can be used to reproduce the
     regression.

The third step can also be used on its own, via the `--passing-container` and
`--failing-container` options, which allows it to be used between private container
tags, without the dependency on the `ghcr.io/nvidia/jax` registry. This assumes that
the given containers are closely related to those from JAX-Toolbox
(`ghcr.io/nvidia/jax:XXX`):
* JAX, XLA, Flax[, MaxText] sources at `/opt/{jax,xla,flax,maxtext}[-source]`
* `build-jax.sh` script from JAX-Toolbox available in the container

## Installation

The triage tool can be installed using `pip`:
```bash
pip install git+https://github.com/NVIDIA/JAX-Toolbox.git#subdirectory=.github/triage
```
or directly from a checkout of the JAX-Toolbox repository.

You should make sure `pip` is up to date, for example with `pip install -U pip`. The
versions of `pip` installed on cluster head/compute nodes can be quite old. The
recommended installation method, using `virtualenv`, should take care of this for you.

Because the tool needs to orchestrate running commands in multiple containers, it is
most convenient to install it in a virtual environment on the host system, rather than
attempting to install it inside a container.

The recommended installation method is to install `virtualenv` natively on the host
system, and then use that to create an isolated environment on the host system for the
triage tool, *i.e.*:
```bash
virtualenv triage-venv
./triage-venv/bin/pip install git+https://github.com/NVIDIA/JAX-Toolbox.git#subdirectory=.github/triage
./triage-venv/bin/jax-toolbox-triage ...
```

The tool should be invoked on a machine with `docker` available and whatever GPUs are
needed to execute the test case if the default runtime (`--container-runtime=docker`)
is used.
If `--container-runtime=pyxis` is used instead, the tool should be invoked on a machine
where `srun --container-image=XXX ... test_command` will execute the test case on one
or more machines with appropriate GPUs, *e.g.* inside an `salloc` session.
Appropriate arguments (number of nodes, number of tasks per node, *etc.*) should be
passed to `salloc` or set via `SLURM_` environment variables so that a bare `srun` will
correctly launch the test case.

## Usage

To use the tool, there are two compulsory inputs:
   * A test command to triage.
   * A specification of which containers to triage in. There are two choices here:
     * `--container`: which of the `ghcr.io/nvidia/jax:CONTAINER-YYYY-MM-DD` container
       families to execute the test command in. Example: `jax` for a JAX unit test
       failure, `maxtext` for a MaxText model execution failure. The `--start-date` and
       `--end-date` options can be combined with `--container` to tune the search; see
       below for more details.
     * `--passing-container` and `--failing-container`: a pair of URLs to containers to
       use in the commit-level search; if these are passed then no container-level
       search is performed.

The test command will be executed directly in the container, not inside a shell, so be
sure not to add excessive quotation marks (*i.e.* run
`jax-toolbox-triage --container=jax test-jax.sh foo` not
`jax-toolbox-triage --container=jax "test-jax.sh foo"`), and you should aim to make it
as fast and targeted as possible.

If you want to run multiple commands, you might want to use something like
`jax-toolbox-triage --container=jax sh -c "command1 && command2"`.

Alternatively, you can use `-v` (`--container-mount`) to mount a host directory
containing test scripts into the container and execute a script from there, *e.g.*
`-v $PWD:/work /work/test.sh`.

The expectation is that the test case will be executed successfully several times as
part of the triage, so you may want to tune some parameters to reduce the execution
time in the successful case.
For example, if `text-maxtext.sh --steps=500 ...` is failing on step 0, you should
probably reduce `--steps` to optimise execution time in the successful case.

A JSON status file and both info-level and debug-level logfiles are written to the
directory given by `--output-prefix`.
Info-level output is also written to the console, and includes the path to the debug
log file.

You should pay attention to the first execution of your test case, to make sure it is
failing for the correct reason. For example:
```console
jax-toolbox-triage --container jax command-you-forgot-to-install
```
will not immediately abort, because the tool is **expecting** the command to fail in
the early stages of the triage:
```
[INFO] 2024-10-29 01:49:01 Verbose output, including stdout/err of triage commands, will be written to /home/olupton/JAX-Toolbox/triage-2024-10-29-01-49-01/debug.log
[INFO] 2024-10-29 01:49:05 Checking end-of-range failure in 2024-10-27
[INFO] 2024-10-29 01:49:05 Ran test case in 2024-10-27 in 0.4s, pass=False
[INFO] 2024-10-29 01:49:05 stdout: OCI runtime exec failed: exec failed: unable to start container process: exec: "command-you-forgot-to-install": executable file not found in $PATH: unknown

[INFO] 2024-10-29 01:49:05 stderr:
[INFO] 2024-10-29 01:49:05 IMPORTANT: you should check that the test output above shows the *expected* failure of your test case in the 2024-10-27 container. It is very easy to accidentally provide a test case that fails for the wrong reason, which will not triage the correct issue!
[INFO] 2024-10-29 01:49:06 Starting coarse search with 2024-10-26 based on end_date=2024-10-27
[INFO] 2024-10-29 01:49:06 Ran test case in 2024-10-26 in 0.4s, pass=False
```
where, notably, the triage search is continuing.

### Optimising container-level search performance

By default, the container-level search starts from the most recent available container,
if you already know that the test has been failing for a while, you can pass
`--end-date` to start the search further in the past.
If you are sure that the test is failing on the `--end-date` you have passed, you can
skip verification of that fact by passing `--skip-precondition-checks` (but see below
for other checks that this skips).

By default, the container-level backwards search for a date on which the test passed
tries the containers approximately [1, 2, 4, ...] days before `--end-date`.
This can be tuned by passing `--start-date`, which overrides the "end date minus one"
start value (but leaves the exponential growth of the search range width).
If you are sure that the test is passing on the `--start-date` you have passed, you can
skip verification of that fact by passing `--skip-precondition-checks`.

The combination of `--start-date`, `--end-date` and `--skip-precondition-checks` can be
used to skip the entire first stage of the bisection process.

The second stage of the triage process can be made to abort early using the
`--threshold-days` option; this stage will terminate once the delta between the latest
known-good and earliest known-bad containers is below the threshold.

If you need to re-start the tool for some reason, use of these options can help
bootstrap the tool using the results of a previous (partial) run.

### Optimising commit-level search performance

The third stage of the triage process involves repeatedly building JAX and XLA, which
can be sped up significantly using a Bazel cache.
By default, a local directory on the host machine (where the tool is being executed)
will be used, but it may be more efficient to use a persistent and/or pre-heated cache.
This can be achieved by passing the `--bazel-cache` option, which accepts absolute
paths and `http`/`https`/`grpc` URLs.

If `--skip-precondition-checks` is passed, a sanity check that the failure can be
reproduced after rebuilding the JAX/XLA commits from the first-known-bad container
inside that container will be skipped. 

## Example

Here is an example execution for a JAX unit test failure, with some annotation (note
that a more modern version of the tool would have tracked Flax too):
```console
user@gpu-machine $ jax-toolbox-triage --container jax test-jax.sh //tests:nn_test_gpu
```
`--end-date` was not passed, and 2024-10-15 is the most recent available container
at the time of execution
```
[INFO] 2024-10-16 00:31:41 Checking end-of-range failure in 2024-10-15
```
`--skip-precondition-checks` was not passed, so the tool checks that the test does, in
fact, fail in the 2024-10-15 container
```
[INFO] 2024-10-16 00:33:36 Ran test case in 2024-10-15 in 114.8s, pass=False
```
`--start-date` was not passed, so the first (backwards search) stage of the triage
process starts with the container 1 day before the end of the range, *i.e.* 2024-10-14
```
[INFO] 2024-10-16 00:33:37 Starting coarse search with 2024-10-14 based on end_date=2024-10-15
[INFO] 2024-10-16 00:35:35 Ran test case in 2024-10-14 in 118.1s, pass=False
```
`end_date - 2 * (end_date - search_date)` = `2024-10-15 - 2 days` = `2024-10-13`
```
[INFO] 2024-10-16 00:38:11 Ran test case in 2024-10-13 in 122.4s, pass=False
```
In principle this would be 4 days before the end date, but the 2024-10-11 container
does not exist, so the tool chooses a nearby container that does exist and is older
than 2024-10-13
```
[INFO] 2024-10-16 00:40:53 Ran test case in 2024-10-12 in 127.7s, pass=False
```
Steps in date start to increase significantly
```
[INFO] 2024-10-16 00:43:28 Ran test case in 2024-10-09 in 119.3s, pass=False
[INFO] 2024-10-16 00:45:29 Ran test case in 2024-10-03 in 120.7s, pass=False
[INFO] 2024-10-16 00:47:27 Ran test case in 2024-09-21 in 116.3s, pass=False
```
The first stage of the triage process successfully identifies an old container where
this test passed
```
[INFO] 2024-10-16 00:51:22 Ran test case in 2024-08-28 in 194.0s, pass=True
[INFO] 2024-10-16 00:51:22 Coarse container-level search yielded [2024-08-28, 2024-09-21]...
```
The second stage of the triage process refines the container-level range by bisection
```
[INFO] 2024-10-16 00:53:19 Ran test case in 2024-09-09 in 115.5s, pass=True
[INFO] 2024-10-16 00:53:19 Refined container-level range to [2024-09-09, 2024-09-21]
[INFO] 2024-10-16 00:56:03 Ran test case in 2024-09-15 in 125.4s, pass=True
[INFO] 2024-10-16 00:56:03 Refined container-level range to [2024-09-15, 2024-09-21]
[INFO] 2024-10-16 00:58:07 Ran test case in 2024-09-18 in 122.9s, pass=True
[INFO] 2024-10-16 00:58:07 Refined container-level range to [2024-09-18, 2024-09-21]
```
The second stage of the triage process converges
```
[INFO] 2024-10-16 01:00:09 Ran test case in 2024-09-19 in 121.2s, pass=False
[INFO] 2024-10-16 01:00:09 Refined container-level range to [2024-09-18, 2024-09-19]
```
The third stage of the triage process begins, using:
 - the first-known-bad container 2024-09-19
 - first-known-bad commits (JAX 9d2e9... and XLA 42b04...)
 - last-known-good commits (JAX 988ed... and XLA 88935...)
```
[INFO] 2024-10-16 01:00:10 Bisecting JAX [988ed2bd75df5fe25b74eaf38075aadff19be207, 9d2e9c688c4e8b733e68467d713091436a672ac0] and XLA [8893550a604fe39aae2eeae49a836e92eed497d1, 42b04a6739dc648a80dd4f3b4e1322f1b2c7f3a7] using ghcr.io/nvidia/jax:jax-2024-09-19
[INFO] 2024-10-16 01:00:10 Building in the range-ending container...
```
Sanity check that re-building the first-known-bad commits in the first-known-bad
container reproduces the failure
```
[INFO] 2024-10-16 01:00:12 Checking out XLA 42b04a6739dc648a80dd4f3b4e1322f1b2c7f3a7 JAX 9d2e9c688c4e8b733e68467d713091436a672ac0
```
No Bazel cache was passed, and this is the first build in the triage session, so it is
slow -- a full rebuild of JAX and XLA was needed
```
[INFO] 2024-10-16 01:13:56 Build completed in 824.9s
[INFO] 2024-10-16 01:15:25 Test completed in 88.5s
[INFO] 2024-10-16 01:15:25 Verified test failure after vanilla rebuild
```
Verification that the last-known-good commits still pass when rebuilt in the
first-known-bad container; this is a bit faster because the Bazel cache is warmer
```
[INFO] 2024-10-16 01:15:25 Checking out XLA 8893550a604fe39aae2eeae49a836e92eed497d1 JAX 988ed2bd75df5fe25b74eaf38075aadff19be207
[INFO] 2024-10-16 01:26:43 Build completed in 677.5s
[INFO] 2024-10-16 01:27:36 Test completed in 53.7s
[INFO] 2024-10-16 01:27:36 Test passed after rebuilding commits from start container in end container
```
Binary search in commits continues, with progressively faster build times
```
[INFO] 2024-10-16 01:27:37 Checking out XLA b976dd94f11ab130c5f718b360fcfb5ac6d6b875 JAX b51c65357f0ae9659e58e2ff0df871542124cddf
[INFO] 2024-10-16 01:32:24 Build completed in 287.7s
[INFO] 2024-10-16 01:33:19 Test completed in 54.4s
[INFO] 2024-10-16 01:33:19 Checking out XLA e291dfe0a12ec5907636a722c545c19d43f04c8b JAX 9dd363da1298e4810b693a918fc2e8199094acdb
[INFO] 2024-10-16 01:34:58 Build completed in 98.9s
[INFO] 2024-10-16 01:35:52 Test completed in 54.1s
[INFO] 2024-10-16 01:35:53 Checking out XLA 6e652a5d91657cfbe9fbcdff4a0ccd1b803675a7 JAX b164d67d4a9bd094426ff450fe1f1335d3071d03
[INFO] 2024-10-16 01:36:54 Build completed in 61.3s
[INFO] 2024-10-16 01:37:47 Test completed in 52.7s
[INFO] 2024-10-16 01:37:47 Checking out XLA a1299f86507c79c8acf877344d545f10329f8515 JAX b164d67d4a9bd094426ff450fe1f1335d3071d03
[INFO] 2024-10-16 01:38:39 Build completed in 52.5s
[INFO] 2024-10-16 01:39:32 Test completed in 52.5s
[INFO] 2024-10-16 01:39:32 Checking out XLA 2d1f7b70740649a57ec4988702ae1dbdfeee6e9c JAX b164d67d4a9bd094426ff450fe1f1335d3071d03
[INFO] 2024-10-16 01:40:24 Build completed in 52.2s
[INFO] 2024-10-16 01:41:17 Test completed in 52.9s
[INFO] 2024-10-16 01:41:17 Checking out XLA 662eb45a17c76df93e5a386929653ae4c1f593da JAX 016c49951f670256ce4750cdfea182e3a2a15325
[INFO] 2024-10-16 01:42:08 Build completed in 50.9s
[INFO] 2024-10-16 01:43:12 Test completed in 64.2s
```
The XLA commit has stopped changing; the initial bisection is XLA-centric (with JAX
kept roughly in sync), but when this converges on a single XLA commit, the tool will
run extra tests to decide whether to blame that XLA commit or a nearby JAX commit
```
[INFO] 2024-10-16 01:43:13 Checking out XLA 662eb45a17c76df93e5a386929653ae4c1f593da JAX b164d67d4a9bd094426ff450fe1f1335d3071d03
[INFO] 2024-10-16 01:44:01 Build completed in 48.8s
[INFO] 2024-10-16 01:45:02 Test completed in 60.8s
[INFO] 2024-10-16 01:45:03 Checking out XLA 662eb45a17c76df93e5a386929653ae4c1f593da JAX cd04d0f32e854aa754e37e4b676725655a94e731
[INFO] 2024-10-16 01:45:52 Build completed in 49.4s
[INFO] 2024-10-16 01:46:53 Test completed in 60.7s
[INFO] 2024-10-16 01:46:53 Bisected failure to JAX cd04d0f32e854aa754e37e4b676725655a94e731..b164d67d4a9bd094426ff450fe1f1335d3071d03 with XLA 662eb45a17c76df93e5a386929653ae4c1f593da
```

Where the final result should be read as saying that the test passes with
[xla@662eb](https://github.com/openxla/xla/commit/662eb45a17c76df93e5a386929653ae4c1f593da)
and
[jax@cd04d](https://github.com/jax-ml/jax/commit/cd04d0f32e854aa754e37e4b676725655a94e731),
but that if JAX is moved forward to include
[jax@b164d](https://github.com/jax-ml/jax/commit/b164d67d4a9bd094426ff450fe1f1335d3071d03)
then the test fails.
This failure is fixed in [jax#24427](https://github.com/jax-ml/jax/pull/24427).

## Other features
The tool will mount a host system directory (under `--output-prefix`) into the
container at `/triage-tool-output` and will make sure that this directory is unique for
each container + commits that is tested. After the triage has converged, the tool will
create `first-known-bad` and `last-known-good` symlinks under `--output-prefix` that
identify the directories corresponding to immediately before and after the problematic
commit.

This can be useful to save metadata, such as HLO dump files or profile data.

**Important**: if your test case launches multiple processes, it is your responsibility
to segregate their output underneath `/triage-tool-output`, for example by using
`$SLURM_PROCID` to only write output from one process, or to write to process-dependent
locations.

## Limitations

This tool aims to target the common case that regressions are due to commits in JAX or
XLA, so if the root cause is different it may not converge, although the partial results
may still be helpful.

For example, if the regression is due to a new version of some other dependency
`SomeProject` that was first installed in the `2024-10-15` container, then the first
two stages of the triage process will correctly identify that `2024-10-15` is the
critical date, but the third stage will fail because it will try and fail to reproduce
test success by building the JAX/XLA commits from `2024-10-14` in the `2024-10-15`
container.

The tool also does not currently handle skipping commits that do not compile.

If you run into these limitations in real-world usage of this tool, please file a bug
against JAX-Toolbox including details of manual steps you took to root-case the test
regression.
