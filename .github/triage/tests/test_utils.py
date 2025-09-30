import logging
import os
import psutil
import signal
import time

from jax_toolbox_triage.utils import run_and_log


def test_run_and_log():
    pid = os.fork()
    if pid == 0:
        # We are the child of the fork
        try:
            run_and_log(["sleep", "30"], logging.getLogger(), stderr="interleaved")
        except KeyboardInterrupt:
            os._exit(42)
    else:
        # We are the parent of the fork
        child = psutil.Process(pid)
        assert child.is_running()
        # Wait until the grandchild `sleep` has appeared
        for _ in range(50):
            children = child.children(recursive=False)
            if len(children):
                break
            time.sleep(0.1)
        else:
            raise Exception(f"Never saw any children of {pid} that we forked")
        # Expect exactly one grandchild
        assert len(children) == 1, children
        (grandchild,) = children
        # Trigger KeyboardInterrupt in the child Python that we forked
        child.send_signal(signal.SIGINT)
        # Child becomes a zombie until we wait on it; we should not get None
        assert child.wait(timeout=5) == 42
        # Wait until the grandchild `sleep` has been killed by SIGTERM in
        # run_and_log, it's possible that we'll be too slow to see the code
        assert grandchild.wait(timeout=5) in {None, -signal.SIGTERM}
