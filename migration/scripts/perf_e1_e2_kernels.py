"""Performance comparison for the 8 rust kernels landed in Wave E1 + E2.

Stand-alone benchmark — does NOT route through the parity_speed harness
because these kernels aren't yet wired into _inputs.GENERATORS. Each
kernel:

  - Inputs generated inline at multiple n values
  - Rust path: in-process call to `adam_core._rust.api`
  - Legacy path: subprocess to `.legacy-venv/bin/python` calling the
    upstream `adam_core` (which still has the JAX/numpy implementations)

Outputs a markdown table to stdout, JSON artifact to
`migration/artifacts/perf_e1_e2_kernels.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_VENV_PYTHON = REPO_ROOT / ".legacy-venv" / "bin" / "python"


def _legacy_call(setup_src: str, call_src: str, kwargs: dict[str, Any], reps: int) -> list[float]:
    """Run a legacy call `reps` times in a fresh subprocess and return per-rep elapsed."""
    payload = pickle.dumps(kwargs, protocol=pickle.HIGHEST_PROTOCOL)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    # The subprocess driver: NOT indented — `python -c` requires column-0
    # statements. The setup_src and call_src snippets are inserted verbatim;
    # callers should write them as column-0 Python.
    script = (
        "import pickle, sys, time\n"
        "kwargs = pickle.load(sys.stdin.buffer)\n"
        f"{setup_src.strip()}\n"
        "_ = call(**kwargs)\n"  # warmup
        "elapsed = []\n"
        f"for _ in range({reps}):\n"
        "    t0 = time.perf_counter()\n"
        f"    {call_src}\n"
        "    elapsed.append(time.perf_counter() - t0)\n"
        "sys.stdout.buffer.write(pickle.dumps(elapsed, protocol=pickle.HIGHEST_PROTOCOL))\n"
    )
    proc = subprocess.run(
        [str(LEGACY_VENV_PYTHON), "-c", script],
        input=payload, capture_output=True, env=env, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"legacy subprocess failed:\n{proc.stderr.decode(errors='replace')}"
        )
    return pickle.loads(proc.stdout)


def _rust_time(fn: Callable, reps: int, **kwargs: Any) -> list[float]:
    fn(**kwargs)  # warmup
    elapsed = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(**kwargs)
        elapsed.append(time.perf_counter() - t0)
    return elapsed


@dataclass
class KernelBench:
    name: str
    rust_call: Callable
    legacy_setup: str
    legacy_call: str
    make_inputs: Callable[[int], dict[str, Any]]


def _percentile(samples: list[float], q: float) -> float:
    return float(np.percentile(samples, q))


def run_kernel(b: KernelBench, n_values: list[int], reps: int) -> list[dict]:
    out = []
    for n in n_values:
        kw = b.make_inputs(n)
        try:
            rust_t = _rust_time(b.rust_call, reps=reps, **kw)
        except Exception as e:
            print(f"  rust err for {b.name}@n={n}: {e}", file=sys.stderr)
            continue
        try:
            leg_t = _legacy_call(b.legacy_setup, b.legacy_call, kw, reps=reps)
        except Exception as e:
            print(f"  legacy err for {b.name}@n={n}: {e}", file=sys.stderr)
            continue
        rust_p50 = _percentile(rust_t, 50)
        leg_p50 = _percentile(leg_t, 50)
        out.append({
            "kernel": b.name,
            "n": n,
            "rust_p50_s": rust_p50,
            "legacy_p50_s": leg_p50,
            "speedup": leg_p50 / rust_p50 if rust_p50 > 0 else float("inf"),
        })
        print(
            f"{b.name:48s} n={n:>7}  "
            f"rust={_fmt(rust_p50):>10}  leg={_fmt(leg_p50):>10}  "
            f"×{out[-1]['speedup']:>5.2f}",
            flush=True,
        )
    return out


def _fmt(t: float) -> str:
    if t >= 1.0:    return f"{t:.2f}s"
    if t >= 1e-3:  return f"{t * 1e3:.2f}ms"
    return f"{t * 1e6:.1f}μs"


# ---------------------------------------------------------------------------
# Per-kernel input generators + legacy call snippets
# ---------------------------------------------------------------------------

MU_SUN = 0.29591220828411956e-3


def make_tisserand_inputs(n: int) -> dict:
    rng = np.random.default_rng(20260427)
    return {
        "a": rng.uniform(0.5, 50.0, size=n),
        "e": rng.beta(2.0, 5.0, size=n) * 0.95,
        "i": rng.uniform(0.0, 175.0, size=n),
        "third_body": "jupiter",
    }


def make_classification_inputs(n: int) -> dict:
    rng = np.random.default_rng(20260427)
    a = rng.uniform(0.5, 50.0, size=n)
    e = rng.beta(2.0, 5.0, size=n) * 0.95
    q = a * (1.0 - e)
    Q = a * (1.0 + e)
    return {"a": a, "e": e, "q": q, "Q": Q}


def make_chi2_inputs(n: int, d: int = 2) -> dict:
    rng = np.random.default_rng(20260427)
    r = rng.standard_normal((n, d)) * 1e-5
    sig = rng.uniform(1e-6, 5e-5, size=n)
    cov = np.zeros((n, d, d))
    for k in range(d):
        cov[:, k, k] = sig ** 2
    return {"residuals": r, "covariances": cov}


def make_weighted_mean_inputs(n: int, d: int = 6) -> dict:
    rng = np.random.default_rng(20260427)
    samples = rng.standard_normal((n, d))
    W = np.full(n, 1.0 / n)
    return {"samples": samples, "W": W}


def make_weighted_cov_inputs(n: int, d: int = 6) -> dict:
    rng = np.random.default_rng(20260427)
    samples = rng.standard_normal((n, d))
    W_cov = np.full(n, 1.0 / n)
    mean = W_cov @ samples
    return {"mean": mean, "samples": samples, "W_cov": W_cov}


def make_fit_abs_mag_inputs(n: int) -> dict:
    rng = np.random.default_rng(20260427)
    return {
        "h_rows": rng.normal(loc=15.0, scale=0.3, size=n),
        "sigma_rows": rng.uniform(0.05, 0.2, size=n),
    }


def make_bound_lon_inputs(n: int) -> dict:
    rng = np.random.default_rng(20260427)
    obs = np.zeros((n, 6))
    obs[:, 1] = rng.uniform(0.0, 360.0, size=n)
    res = np.zeros((n, 6))
    # Inject some boundary-crossing residuals
    res[:, 1] = rng.uniform(-360.0, 360.0, size=n)
    return {"observed": obs, "residuals": res}


def make_cos_lat_inputs(n: int) -> dict:
    rng = np.random.default_rng(20260427)
    lat = rng.uniform(-89.0, 89.0, size=n)
    res = rng.standard_normal((n, 6)) * 1e-3
    cov = np.zeros((n, 6, 6))
    for k in range(6):
        cov[:, k, k] = 1e-6
    return {"lat": lat, "residuals": res, "covariances": cov}


# ---------------------------------------------------------------------------

# Rust call wrappers - import lazily so the script stays portable
def _rust_tisserand(**kw):
    from adam_core.dynamics.tisserand import calc_tisserand_parameter
    return calc_tisserand_parameter(**kw)

def _rust_classification(**kw):
    from adam_core._rust.api import classify_orbits_numpy
    a = np.ascontiguousarray(kw["a"], dtype=np.float64)
    e = np.ascontiguousarray(kw["e"], dtype=np.float64)
    q = np.ascontiguousarray(kw["q"], dtype=np.float64)
    Q = np.ascontiguousarray(kw["Q"], dtype=np.float64)
    return classify_orbits_numpy(a, e, q, Q)

def _rust_chi2(**kw):
    from adam_core._rust.api import calculate_chi2_numpy
    return calculate_chi2_numpy(kw["residuals"], kw["covariances"])

def _rust_weighted_mean(**kw):
    from adam_core._rust.api import weighted_mean_numpy
    return weighted_mean_numpy(kw["samples"], kw["W"])

def _rust_weighted_cov(**kw):
    from adam_core._rust.api import weighted_covariance_numpy
    return weighted_covariance_numpy(kw["mean"], kw["samples"], kw["W_cov"])

def _rust_fit_abs_mag(**kw):
    from adam_core._rust.api import fit_absolute_magnitude_rows_numpy
    return fit_absolute_magnitude_rows_numpy(kw["h_rows"], kw["sigma_rows"])

def _rust_bound_lon(**kw):
    from adam_core._rust.api import bound_longitude_residuals_numpy
    return bound_longitude_residuals_numpy(kw["observed"], kw["residuals"])

def _rust_cos_lat(**kw):
    from adam_core._rust.api import apply_cosine_latitude_correction_numpy
    return apply_cosine_latitude_correction_numpy(kw["lat"], kw["residuals"], kw["covariances"])


KERNELS = [
    KernelBench(
        name="dynamics.calc_tisserand_parameter",
        rust_call=_rust_tisserand,
        legacy_setup=textwrap.dedent("""
            from adam_core.dynamics.tisserand import calc_tisserand_parameter as _legacy_fn
            def call(**kw):
                return _legacy_fn(**kw)
        """),
        legacy_call="call(**kwargs)",
        make_inputs=make_tisserand_inputs,
    ),
    KernelBench(
        name="orbits.classification (numpy core)",
        rust_call=_rust_classification,
        # The legacy `calc_orbit_class` takes a quivr KeplerianCoordinates
        # table, but the rust path receives flat arrays. We benchmark the
        # *equivalent numpy work* on the legacy side: the np.where chain
        # over (a, e, q, Q). This isolates the numerical-classification
        # cost from the quivr table-construction cost (which is shared).
        legacy_setup=textwrap.dedent("""
            import numpy as np
            def call(a, e, q, Q):
                cls = np.array(['AST'] * len(a), dtype=object)
                rules = {
                    'AMO': np.where((a > 1.0) & (q > 1.017) & (q < 1.3)),
                    'APO': np.where((a > 1.0) & (q < 1.017)),
                    'ATE': np.where((a < 1.0) & (Q > 0.983)),
                    'CEN': np.where((a > 5.5) & (a < 30.1)),
                    'IEO': np.where((Q < 0.983)),
                    'IMB': np.where((a < 2.0) & (q > 1.666)),
                    'MBA': np.where((a > 2.0) & (a < 3.2) & (q > 1.666)),
                    'MCA': np.where((a < 3.2) & (q > 1.3) & (q < 1.666)),
                    'OMB': np.where((a > 3.2) & (a < 4.6)),
                    'TJN': np.where((a > 4.6) & (a < 5.5) & (e < 0.3)),
                    'TNO': np.where((a > 30.1)),
                    'PAA': np.where((e == 1)),
                    'HYA': np.where((e > 1)),
                }
                for c, idx in rules.items():
                    cls[idx] = c
                return cls
        """),
        legacy_call="call(**kwargs)",
        make_inputs=make_classification_inputs,
    ),
    KernelBench(
        name="coordinates.residuals.calculate_chi2 (D=2)",
        rust_call=_rust_chi2,
        legacy_setup=textwrap.dedent("""
            import numpy as np
            def call(residuals, covariances):
                W = np.linalg.inv(covariances)
                return np.einsum('ij,ji->i', np.einsum('ij,ijk->ik', residuals, W), residuals.T)
        """),
        legacy_call="call(**kwargs)",
        make_inputs=lambda n: make_chi2_inputs(n, d=2),
    ),
    KernelBench(
        name="coordinates.covariances.weighted_mean",
        rust_call=_rust_weighted_mean,
        legacy_setup=textwrap.dedent("""
            import numpy as np
            def call(samples, W):
                return np.dot(W, samples)
        """),
        legacy_call="call(**kwargs)",
        make_inputs=make_weighted_mean_inputs,
    ),
    KernelBench(
        name="coordinates.covariances.weighted_covariance",
        rust_call=_rust_weighted_cov,
        legacy_setup=textwrap.dedent("""
            import numpy as np
            def call(mean, samples, W_cov):
                residual = samples - mean
                return (W_cov * residual.T) @ residual
        """),
        legacy_call="call(**kwargs)",
        make_inputs=make_weighted_cov_inputs,
    ),
    KernelBench(
        name="photometry.absolute_magnitude._fit_absolute_magnitude_rows",
        rust_call=_rust_fit_abs_mag,
        legacy_setup=textwrap.dedent("""
            import numpy as np
            def _mad(x):
                med = float(np.nanmedian(x))
                mad = float(np.nanmedian(np.abs(x - med)))
                return 1.4826 * mad
            def call(h_rows, sigma_rows):
                n = len(h_rows)
                have_all = bool(np.all(np.isfinite(sigma_rows)))
                if have_all:
                    w = 1.0 / np.square(sigma_rows)
                    H = float(np.sum(w * h_rows) / np.sum(w))
                else:
                    H = float(np.mean(h_rows))
                resid = h_rows - H
                sigma_eff = float(_mad(resid)) if n >= 2 else None
                if have_all and n >= 2:
                    w = 1.0 / np.square(sigma_rows)
                    chi2_red = float(np.sum(w * np.square(resid))) / (n - 1)
                    H_sigma = float(np.sqrt(1.0 / np.sum(w)))
                    if chi2_red > 1.0:
                        H_sigma *= np.sqrt(chi2_red)
                else:
                    H_sigma = None; chi2_red = None
                return (H, H_sigma, sigma_eff, chi2_red, n)
        """),
        legacy_call="call(**kwargs)",
        make_inputs=make_fit_abs_mag_inputs,
    ),
    KernelBench(
        name="coordinates.residuals.bound_longitude_residuals",
        rust_call=_rust_bound_lon,
        legacy_setup=textwrap.dedent("""
            import numpy as np
            def call(observed, residuals):
                lon_obs = observed[:, 1]
                lr = residuals[:, 1].copy()
                g180 = lr > 180; l180 = lr < -180
                lr = np.where(g180, lr - 360, lr)
                lr = np.where(l180, lr + 360, lr)
                lr = np.where(g180 & (lon_obs > 180), -lr, lr)
                lr = np.where(l180 & (lon_obs < 180), -lr, lr)
                out = residuals.copy()
                out[:, 1] = lr
                return out
        """),
        legacy_call="call(**kwargs)",
        make_inputs=make_bound_lon_inputs,
    ),
    KernelBench(
        name="coordinates.residuals.apply_cosine_latitude_correction",
        rust_call=_rust_cos_lat,
        legacy_setup=textwrap.dedent("""
            import numpy as np
            def call(lat, residuals, covariances):
                cos_lat = np.cos(np.radians(lat))
                identity = np.identity(6, dtype=np.float64)
                diag = np.ones((len(lat), 6))
                diag[:, 1] = cos_lat; diag[:, 4] = cos_lat
                cos_lat_cov = np.einsum('kj,ji->kij', diag, identity, order='C')
                res = residuals.copy()
                res[:, 1] *= cos_lat; res[:, 4] *= cos_lat
                nan_cov = np.isnan(covariances)
                cov_m = np.where(nan_cov, 0.0, covariances)
                cov_m = cos_lat_cov @ cov_m @ cos_lat_cov.transpose(0, 2, 1)
                cov = np.where(nan_cov, np.nan, cov_m)
                return (res, cov)
        """),
        legacy_call="call(**kwargs)",
        make_inputs=make_cos_lat_inputs,
    ),
]


def _format_table(rows: list[dict]) -> str:
    by_kernel: dict[str, list[dict]] = {}
    for r in rows:
        by_kernel.setdefault(r["kernel"], []).append(r)
    ns = sorted({r["n"] for r in rows})
    lines = []
    header = ["| API |"] + [f" n={n} |" for n in ns]
    lines.append("".join(header))
    lines.append("| --- |" + " ---: |" * len(ns))
    for kernel in by_kernel:
        cells = [f"| `{kernel}` |"]
        for n in ns:
            cell = next((r for r in by_kernel[kernel] if r["n"] == n), None)
            if cell is None:
                cells.append(" — |")
            else:
                cells.append(
                    f" {_fmt(cell['rust_p50_s'])} / "
                    f"{_fmt(cell['legacy_p50_s'])} / "
                    f"{cell['speedup']:.2f}× |"
                )
        lines.append("".join(cells))
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ns", nargs="*", type=int, default=[10, 100, 1000, 10000, 100000])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--output", type=Path,
                   default=Path("migration/artifacts/perf_e1_e2_kernels.json"))
    args = p.parse_args(argv)

    all_rows: list[dict] = []
    for kernel in KERNELS:
        print(f"=== {kernel.name} ===", file=sys.stderr)
        all_rows.extend(run_kernel(kernel, args.ns, args.reps))

    print()
    print("# Wave E1 + E2 rust kernel performance comparison\n")
    print(f"Workload: {args.reps} reps after 1 warmup. Cell: rust / legacy / speedup at each N.\n")
    print(_format_table(all_rows))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(all_rows, indent=2))
    print(f"\nwrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
