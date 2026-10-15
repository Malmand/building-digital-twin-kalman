"""Cost of running the twin: time per 5-min update, time per 6-h forecast, and memory.

Measured with the most detailed model (M3, five augmented states) on the machine the script runs on;
results/benchmark.json records the platform as well.

    python tools/benchmark.py
"""
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from twin.rc import RCModel  # noqa: E402
from twin.twin import DigitalTwin  # noqa: E402


def main(n=20000):
    m = RCModel("M3")
    rng = np.random.default_rng(0)
    # synthetic inputs: timing only
    U = np.c_[30 + rng.standard_normal(n), rng.uniform(0, 5, n), rng.uniform(0, 2, n), rng.uniform(0, 0.9, n), np.ones(n)]
    Y = 25 + 0.3 * rng.standard_normal(n)

    tw = DigitalTwin(m, 25.0)
    t0 = time.perf_counter()
    for k in range(n):
        tw.step(Y[k], U[k])
    t_step = (time.perf_counter() - t0) / n

    # memory, measured separately because tracing slows the code down
    tracemalloc.start()
    tw2 = DigitalTwin(m, 25.0)
    snap_mem = tracemalloc.get_traced_memory()[0]
    for k in range(2000):
        tw2.step(Y[k], U[k])
    peak_during = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    t0 = time.perf_counter()
    reps = 200
    for _ in range(reps):
        tw.forecast(U[:72])
    t_fc = (time.perf_counter() - t0) / reps

    state_bytes = sum(a.nbytes for a in (tw.x, tw.P, tw.Q, tw.Ad, tw.Bd, tw.b_air, tw.C))
    out = dict(platform=platform.platform(), processor=platform.processor() or platform.machine(),
               python=platform.python_version(), model="M3 (3 states + 2 tracked parameters)",
               step_us=round(t_step * 1e6, 1), forecast_6h_ms=round(t_fc * 1e3, 2),
               state_bytes=int(state_bytes), python_heap_after_init_kB=round(snap_mem / 1e3, 1),
               python_heap_peak_while_running_kB=round(peak_during / 1e3, 1),
               steps_per_day=288, cpu_seconds_per_zone_day=round(t_step * 288 + t_fc * 288 / 12, 3))
    Path(__file__).resolve().parents[1].joinpath("results", "benchmark.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
