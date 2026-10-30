import os, subprocess, sys


def test_importing_bench_tolerates_a_notebook_plotting_backend():
    env = dict(os.environ, MPLBACKEND="module://matplotlib_inline.backend_inline")
    done = subprocess.run([sys.executable, "-c", "import learned_selection.bench"], env=env, capture_output=True,
                          text=True)
    assert done.returncode == 0, done.stderr
