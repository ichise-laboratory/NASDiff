import contextlib
import os


@contextlib.contextmanager
def suppress_third_party_output():
    """Suppress noisy third-party stdout/stderr during setup steps."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
