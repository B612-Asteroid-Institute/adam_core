"""Lane-specific speed gate for Rust vs baseline main at p50 and p95.

For each measured API, time both the current Rust path (in-process) and the
baseline-main path (in the legacy-venv subprocess) on identical workloads, and
assert that each enforced lane meets its configured p50/p95 threshold. The
historical small-n lane keeps the standard 1.2x promotion threshold; tiny-n and
large-n are enforced at the same 1.2x threshold.

Each timing loop runs ``reps`` repetitions inside its respective process so
subprocess invocation overhead is excluded from the legacy measurements.

Warm p50/p95 comparisons default to single process / single thread on both
sides. This is a port-quality gate, not a "Rust used more cores" gate.
Cold-call timings remain process-realistic by default and are labeled with a
separate thread mode in the emitted artifact.

RM-P1-019 adds named size lanes. The historical ``small-n`` lane keeps the
canonical ``n=2000`` pass/fail promotion gate. RM-P1-019A makes the large
API-shaped lane part of enforced governance, adds a tiny one-off lane for
quick-call behavior, and records structured workload axes instead of only a
flat free-form size string.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import _threading

DEFAULT_SMALL_LANE_NAME = "small-n"
DEFAULT_LARGE_LANE_NAME = "large-n"
DEFAULT_TINY_LANE_NAME = "tiny-n"
DEFAULT_TINY_N = 10
DEFAULT_SMALL_N = 2000
# Per `decisions.md` (2026-05-04 18:36 UTC) every lane, especially large
# workloads, must meet at least 1.2x p50/p95 versus the legacy baseline. The
# tiny lane is held to the same 1.2x bar as a quick-call gate.
DEFAULT_TINY_SPEEDUP = 1.2
DEFAULT_SMALL_SPEEDUP = 1.2
DEFAULT_LARGE_SPEEDUP = 1.2
DEFAULT_LEGACY_CACHE_PATH = Path(
    "migration/artifacts/parity_legacy_speed_baseline.json"
)
LEGACY_TIMING_CACHE_SCHEMA_VERSION = 1
LEGACY_TIMING_CACHE_PROCESS_VERSION = "rm-p1-019a-shaped-lanes-v1"
LEGACY_REPO_ROOT = Path("/Users/aleck/Code/adam-core")


@dataclass(frozen=True)
class SpeedLane:
    name: str
    description: str
    enforced: bool
    reps: int
    warmup: int
    measure_cold: bool
    min_speedup_p50: float
    min_speedup_p95: float
    api_workloads: Mapping[str, Any] = field(default_factory=dict)

    def workload_for(self, api_id: str) -> Any:
        try:
            return self.api_workloads[api_id]
        except KeyError as exc:
            raise KeyError(f"No {self.name!r} workload defined for {api_id!r}") from exc


def _percentile(samples: list[float], q: float) -> float:
    if not samples:
        return float("inf")
    import numpy as np

    return float(np.percentile(samples, q))


def _split_api_field(api_field: object) -> list[str]:
    if isinstance(api_field, str):
        return [api.strip() for api in api_field.split(",") if api.strip()]
    if isinstance(api_field, list):
        return [str(api).strip() for api in api_field if str(api).strip()]
    return []


def _review_date(entry: Mapping[str, object]) -> date:
    review_by = entry.get("review_by")
    if isinstance(review_by, date):
        return review_by
    if isinstance(review_by, str):
        return date.fromisoformat(review_by)
    raise ValueError(f"active waiver {entry.get('id')!r} is missing review_by")


def _perf_waivers_by_api_lane() -> dict[tuple[str, str], str]:
    """Return active, unexpired performance waivers keyed by ``(api_id, lane)``.

    Legacy registry waivers are treated as lane-agnostic for backward
    compatibility, but current benchmark governance records lane-specific
    waivers in ``migration/waivers.yaml``. Waiver YAML must parse successfully;
    otherwise the governance gate should fail loudly instead of silently losing
    lane-scoped waiver state.
    """
    from adam_core._rust.status import API_MIGRATIONS

    waivers: dict[tuple[str, str], str] = {
        (migration.api_id, "*"): migration.waiver
        for migration in API_MIGRATIONS
        if migration.waiver
    }

    import yaml

    waiver_path = Path(__file__).resolve().parents[1] / "waivers.yaml"
    data = yaml.safe_load(waiver_path.read_text()) if waiver_path.exists() else {}
    today = date.today()

    for entry in data.get("waivers", []) or []:
        if entry.get("status") != "active":
            continue
        if _review_date(entry) < today:
            continue
        lane = str(entry.get("lane") or "*")
        waiver_id = str(entry.get("id") or "")
        if not waiver_id:
            continue
        for api_id in _split_api_field(entry.get("api")):
            waivers[(api_id, lane)] = waiver_id
    return waivers


def _shape_rows(workload: Any) -> int:
    if hasattr(workload, "rows"):
        return int(workload.rows)
    return int(workload)


def _shape_label(workload: Any) -> str:
    if hasattr(workload, "label"):
        return str(workload.label())
    return f"rows={int(workload)}"


def _shape_json(workload: Any) -> dict[str, object]:
    if hasattr(workload, "to_json"):
        return dict(workload.to_json())
    rows = int(workload)
    return {"rows": rows, "axes": {}, "label": f"rows={rows}"}


def _time_rust(api_id: str, kwargs: dict, *, reps: int, warmup: int) -> list[float]:
    # Import after thread caps are applied so Rayon sees RAYON_NUM_THREADS=1 in
    # the default gate before any parallel iterator initializes its pool.
    from . import _rust_runner

    for _ in range(warmup):
        _rust_runner.run(api_id, **kwargs)
    samples: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _rust_runner.run(api_id, **kwargs)
        samples.append(time.perf_counter() - t0)
    return samples


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_json(value: object) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _benchmark_source_hash() -> str:
    parity_dir = Path(__file__).resolve().parent
    paths = [
        parity_dir / "_inputs.py",
        parity_dir / "_legacy_runner.py",
        parity_dir / "_oracle.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _subprocess_output(args: Sequence[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        list(args),
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.stdout.strip() or proc.stderr.strip()


def _git_output(args: Sequence[str], *, cwd: Path) -> str:
    return _subprocess_output(["git", *args], cwd=cwd)


def _legacy_identity() -> dict[str, object]:
    from . import _oracle

    dirty = bool(
        _git_output(
            ["status", "--porcelain", "--untracked-files=no"], cwd=LEGACY_REPO_ROOT
        )
    )
    return {
        "repo_root": str(LEGACY_REPO_ROOT),
        "git_commit": _git_output(["rev-parse", "HEAD"], cwd=LEGACY_REPO_ROOT),
        "git_dirty": dirty,
        "python": str(_oracle.LEGACY_VENV_PYTHON),
        "python_version": _subprocess_output(
            [str(_oracle.LEGACY_VENV_PYTHON), "--version"]
        ),
        "benchmark_source_hash": _benchmark_source_hash(),
        "process_version": LEGACY_TIMING_CACHE_PROCESS_VERSION,
    }


def _empty_legacy_cache(identity: Mapping[str, object]) -> dict[str, object]:
    now = _utc_now()
    return {
        "schema_version": LEGACY_TIMING_CACHE_SCHEMA_VERSION,
        "process_version": LEGACY_TIMING_CACHE_PROCESS_VERSION,
        "created_at": now,
        "updated_at": now,
        "legacy_identity": dict(identity),
        "warm": {},
        "cold": {},
    }


def _legacy_identity_without_source_hash(
    identity: Mapping[str, object],
) -> dict[str, object]:
    return {k: v for k, v in identity.items() if k != "benchmark_source_hash"}


def _validate_legacy_cache(
    data: Mapping[str, object],
    identity: Mapping[str, object],
    path: Path,
    *,
    allow_source_hash_mismatch: bool = False,
) -> None:
    if data.get("schema_version") != LEGACY_TIMING_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Legacy timing cache {path} has schema_version={data.get('schema_version')}; "
            f"expected {LEGACY_TIMING_CACHE_SCHEMA_VERSION}. Run with "
            "--refresh-legacy-cache to recapture baseline timings."
        )
    if data.get("process_version") != LEGACY_TIMING_CACHE_PROCESS_VERSION:
        raise ValueError(
            f"Legacy timing cache {path} has process_version={data.get('process_version')}; "
            f"expected {LEGACY_TIMING_CACHE_PROCESS_VERSION}. Run with "
            "--replace-legacy-cache after benchmark process changes."
        )
    cached_identity = data.get("legacy_identity")
    if not isinstance(cached_identity, dict):
        raise ValueError(f"Legacy timing cache {path} is missing legacy_identity")
    if cached_identity == dict(identity):
        return
    if allow_source_hash_mismatch and _legacy_identity_without_source_hash(
        cached_identity
    ) == _legacy_identity_without_source_hash(identity):
        return
    raise ValueError(
        f"Legacy timing cache {path} was captured for a different legacy "
        "checkout, Python, or benchmark source hash. Run with "
        "--refresh-legacy-cache to recapture source-hash-only drift, or "
        "--replace-legacy-cache after baseline/process changes."
    )


def prepare_legacy_timing_cache(
    path: Path | None,
    *,
    refresh: bool = False,
    replace: bool = False,
) -> dict[str, object] | None:
    if path is None:
        return None
    identity = _legacy_identity()
    if refresh and replace:
        data = _empty_legacy_cache(identity)
    elif path.exists():
        data = json.loads(path.read_text())
        _validate_legacy_cache(
            data,
            identity,
            path,
            allow_source_hash_mismatch=refresh,
        )
        if refresh:
            data["legacy_identity"] = dict(identity)
    elif refresh:
        data = _empty_legacy_cache(identity)
    else:
        raise FileNotFoundError(
            f"Legacy timing cache {path} does not exist. Run once with "
            "--refresh-legacy-cache to serialize baseline-main timings."
        )
    return {
        "path": path,
        "data": data,
        "refresh": refresh,
        "replace": replace,
        "dirty": False,
        "hits": {"warm": 0, "cold": 0},
        "misses": {"warm": 0, "cold": 0},
        "writes": {"warm": 0, "cold": 0},
    }


def _legacy_cache_fields(
    *,
    kind: str,
    api_id: str,
    lane: str,
    workload_shape: Mapping[str, object],
    seed: int,
    thread_mode: str,
    reps: int | None = None,
    warmup: int | None = None,
) -> dict[str, object]:
    fields: dict[str, object] = {
        "kind": kind,
        "api_id": api_id,
        "lane": lane,
        "workload_shape": dict(workload_shape),
        "seed": seed,
        "thread_mode": thread_mode,
        "process_version": LEGACY_TIMING_CACHE_PROCESS_VERSION,
    }
    if reps is not None:
        fields["reps"] = reps
    if warmup is not None:
        fields["warmup"] = warmup
    return fields


def _cache_data(context: Mapping[str, object]) -> dict[str, object]:
    data = context["data"]
    if not isinstance(data, dict):
        raise TypeError("legacy timing cache context has invalid data")
    return data


def _cache_section(context: Mapping[str, object], section: str) -> dict[str, object]:
    entries = _cache_data(context)[section]
    if not isinstance(entries, dict):
        raise TypeError(f"legacy timing cache section {section!r} is invalid")
    return entries


def _stats_dict(context: Mapping[str, object], name: str) -> dict[str, int]:
    stats = context[name]
    if not isinstance(stats, dict):
        raise TypeError(f"legacy timing cache stats {name!r} is invalid")
    return stats


def _cached_entry(
    context: dict[str, object], section: str, key: str, fields: Mapping[str, object]
) -> Mapping[str, object] | None:
    if bool(context["refresh"]):
        _stats_dict(context, "misses")[section] += 1
        return None
    entry = _cache_section(context, section).get(key)
    if entry is None:
        _stats_dict(context, "misses")[section] += 1
        return None
    if not isinstance(entry, dict):
        raise TypeError(f"legacy timing cache entry {key} is invalid")
    if entry.get("key_fields") != dict(fields):
        raise ValueError(f"legacy timing cache key collision for {key}")
    _stats_dict(context, "hits")[section] += 1
    return entry


def _write_cache_entry(
    context: dict[str, object], section: str, key: str, entry: Mapping[str, object]
) -> None:
    _cache_section(context, section)[key] = dict(entry)
    context["dirty"] = True
    _stats_dict(context, "writes")[section] += 1


def _cache_miss_message(api_id: str, lane: str, workload_label: str, key: str) -> str:
    return (
        f"legacy timing cache miss for {api_id} lane={lane} shape={workload_label} "
        f"key={key}. Run once with --refresh-legacy-cache after benchmark API, "
        "shape, or process changes."
    )


def _time_legacy_warm(
    api_id: str,
    kwargs: dict[str, Any],
    *,
    reps: int,
    warmup: int,
    seed: int,
    thread_mode: str,
    lane: str,
    workload_shape: Mapping[str, object],
    workload_label: str,
    legacy_cache: dict[str, object] | None,
) -> tuple[list[float], str, str]:
    from . import _oracle

    if legacy_cache is None:
        return (
            _oracle.time_legacy(
                api_id,
                reps=reps,
                warmup=warmup,
                thread_mode=thread_mode,
                **kwargs,
            ),
            "measured",
            "",
        )

    fields = _legacy_cache_fields(
        kind="warm",
        api_id=api_id,
        lane=lane,
        workload_shape=workload_shape,
        seed=seed,
        thread_mode=thread_mode,
        reps=reps,
        warmup=warmup,
    )
    key = _hash_json(fields)
    entry = _cached_entry(legacy_cache, "warm", key, fields)
    if entry is not None:
        return [float(value) for value in entry["samples_s"]], "cache", key
    if not bool(legacy_cache["refresh"]):
        raise KeyError(_cache_miss_message(api_id, lane, workload_label, key))

    samples = _oracle.time_legacy(
        api_id,
        reps=reps,
        warmup=warmup,
        thread_mode=thread_mode,
        **kwargs,
    )
    _write_cache_entry(
        legacy_cache,
        "warm",
        key,
        {
            "key_fields": fields,
            "samples_s": samples,
            "p50_s": _percentile(samples, 50),
            "p95_s": _percentile(samples, 95),
            "captured_at": _utc_now(),
        },
    )
    return samples, "refreshed", key


def _time_legacy_cold(
    api_id: str,
    kwargs: dict[str, Any],
    *,
    seed: int,
    cold_thread_mode: str,
    lane: str,
    workload_shape: Mapping[str, object],
    workload_label: str,
    legacy_cache: dict[str, object] | None,
) -> tuple[float, str, str]:
    from . import _oracle

    if legacy_cache is None:
        return (
            _oracle.time_legacy_cold(api_id, thread_mode=cold_thread_mode, **kwargs),
            "measured",
            "",
        )

    fields = _legacy_cache_fields(
        kind="cold",
        api_id=api_id,
        lane=lane,
        workload_shape=workload_shape,
        seed=seed,
        thread_mode=cold_thread_mode,
    )
    key = _hash_json(fields)
    entry = _cached_entry(legacy_cache, "cold", key, fields)
    if entry is not None:
        return float(entry["elapsed_s"]), "cache", key
    if not bool(legacy_cache["refresh"]):
        raise KeyError(_cache_miss_message(api_id, lane, workload_label, key))

    elapsed = _oracle.time_legacy_cold(api_id, thread_mode=cold_thread_mode, **kwargs)
    _write_cache_entry(
        legacy_cache,
        "cold",
        key,
        {"key_fields": fields, "elapsed_s": elapsed, "captured_at": _utc_now()},
    )
    return elapsed, "refreshed", key


def write_legacy_timing_cache(context: Mapping[str, object] | None) -> None:
    if context is None or not bool(context["dirty"]):
        return
    data = _cache_data(context)
    data["updated_at"] = _utc_now()
    path = context["path"]
    if not isinstance(path, Path):
        raise TypeError("legacy timing cache context has invalid path")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _legacy_cache_metadata(
    context: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if context is None:
        return None
    data = _cache_data(context)
    path = context["path"]
    if not isinstance(path, Path):
        raise TypeError("legacy timing cache context has invalid path")
    return {
        "path": str(path),
        "refresh": bool(context["refresh"]),
        "schema_version": data.get("schema_version"),
        "process_version": data.get("process_version"),
        "legacy_identity": data.get("legacy_identity"),
        "hits": dict(_stats_dict(context, "hits")),
        "misses": dict(_stats_dict(context, "misses")),
        "writes": dict(_stats_dict(context, "writes")),
    }


@dataclass
class SpeedResult:
    api_id: str
    n: int
    rust_p50: float
    rust_p95: float
    legacy_p50: float
    legacy_p95: float
    speedup_p50: float
    speedup_p95: float
    raw_passed: bool
    passed: bool
    min_speedup_p50: float = DEFAULT_SMALL_SPEEDUP
    min_speedup_p95: float = DEFAULT_SMALL_SPEEDUP
    waived: bool = False
    waiver: str = ""
    error: Optional[str] = None
    # Size-lane metadata.
    lane: str = DEFAULT_SMALL_LANE_NAME
    lane_description: str = "historical n=2000 promotion gate"
    lane_enforced: bool = True
    workload_shape: dict[str, object] = field(default_factory=dict)
    workload_label: str = ""
    reps: int = 7
    warmup: int = 1
    # Warm p50/p95 thread metadata.
    thread_mode: str = "single"
    thread_env: dict[str, str | None] = field(default_factory=dict)
    legacy_source: str = "measured"
    legacy_cache_key: str = ""
    # Cold-call (one-shot, includes process spawn + import + first call).
    rust_cold: Optional[float] = None
    legacy_cold: Optional[float] = None
    speedup_cold: Optional[float] = None
    cold_thread_mode: str = "multi-thread"
    cold_thread_env: dict[str, str | None] = field(default_factory=dict)
    legacy_cold_source: str = "not-measured"
    legacy_cold_cache_key: str = ""


def _passes(
    legacy_p50: float,
    legacy_p95: float,
    rust_p50: float,
    rust_p95: float,
    *,
    min_speedup_p50: float,
    min_speedup_p95: float,
) -> tuple[bool, float, float]:
    s50 = legacy_p50 / rust_p50 if rust_p50 > 0 else float("inf")
    s95 = legacy_p95 / rust_p95 if rust_p95 > 0 else float("inf")
    passed = s50 >= min_speedup_p50 and s95 >= min_speedup_p95
    return passed, s50, s95


def _cold_env_snapshot(cold_thread_mode: str) -> dict[str, str | None]:
    cold_env = _threading.env_for_thread_mode(cold_thread_mode)
    return _threading.snapshot_thread_env(cold_env)


def _error_result(
    api_id: str,
    n: int,
    error: str,
    *,
    lane: str,
    lane_description: str,
    lane_enforced: bool,
    workload_shape: dict[str, object],
    workload_label: str,
    min_speedup_p50: float,
    min_speedup_p95: float,
    reps: int,
    warmup: int,
    thread_mode: str,
    thread_env: dict[str, str | None],
    cold_thread_mode: str,
    cold_thread_env: dict[str, str | None],
) -> SpeedResult:
    return SpeedResult(
        api_id=api_id,
        n=n,
        rust_p50=float("inf"),
        rust_p95=float("inf"),
        legacy_p50=float("inf"),
        legacy_p95=float("inf"),
        speedup_p50=0.0,
        speedup_p95=0.0,
        raw_passed=False,
        passed=False,
        min_speedup_p50=min_speedup_p50,
        min_speedup_p95=min_speedup_p95,
        error=error,
        lane=lane,
        lane_description=lane_description,
        lane_enforced=lane_enforced,
        workload_shape=workload_shape,
        workload_label=workload_label,
        reps=reps,
        warmup=warmup,
        thread_mode=thread_mode,
        thread_env=thread_env,
        cold_thread_mode=cold_thread_mode,
        cold_thread_env=cold_thread_env,
    )


def measure(
    api_id: str,
    *,
    n: int,
    reps: int = 7,
    warmup: int = 1,
    seed: int = 20260425,
    measure_cold: bool = False,
    thread_mode: str = "single",
    cold_thread_mode: str = "multi-thread",
    lane: str = DEFAULT_SMALL_LANE_NAME,
    lane_description: str = "historical n=2000 promotion gate",
    lane_enforced: bool = True,
    workload: Any | None = None,
    min_speedup_p50: float = DEFAULT_SMALL_SPEEDUP,
    min_speedup_p95: float = DEFAULT_SMALL_SPEEDUP,
    legacy_cache: dict[str, object] | None = None,
) -> SpeedResult:
    thread_mode = _threading.validate_thread_mode(thread_mode)
    cold_thread_mode = _threading.validate_thread_mode(cold_thread_mode)
    thread_env = _threading.apply_thread_mode(thread_mode)
    cold_thread_env = _cold_env_snapshot(cold_thread_mode)
    if workload is None:
        workload = n
    workload_rows = _shape_rows(workload)
    workload_shape = _shape_json(workload)
    workload_label = _shape_label(workload)

    import numpy as np

    from . import _inputs, _oracle

    legacy_source = "measured"
    legacy_cache_key = ""

    rng = np.random.default_rng(seed)
    try:
        sample = _inputs.make(api_id, rng, workload)
    except Exception as e:
        return _error_result(
            api_id,
            workload_rows,
            f"input gen: {type(e).__name__}: {e}",
            lane=lane,
            lane_description=lane_description,
            lane_enforced=lane_enforced,
            workload_shape=workload_shape,
            workload_label=workload_label,
            min_speedup_p50=min_speedup_p50,
            min_speedup_p95=min_speedup_p95,
            reps=reps,
            warmup=warmup,
            thread_mode=thread_mode,
            thread_env=thread_env,
            cold_thread_mode=cold_thread_mode,
            cold_thread_env=cold_thread_env,
        )

    try:
        rust_times = _time_rust(api_id, sample.rust_kwargs, reps=reps, warmup=warmup)
        legacy_times, legacy_source, legacy_cache_key = _time_legacy_warm(
            api_id,
            sample.legacy_kwargs,
            reps=reps,
            warmup=warmup,
            seed=seed,
            thread_mode=thread_mode,
            lane=lane,
            workload_shape=workload_shape,
            workload_label=workload_label,
            legacy_cache=legacy_cache,
        )
    except Exception as e:
        return _error_result(
            api_id,
            workload_rows,
            f"timing: {type(e).__name__}: {e}",
            lane=lane,
            lane_description=lane_description,
            lane_enforced=lane_enforced,
            workload_shape=workload_shape,
            workload_label=workload_label,
            min_speedup_p50=min_speedup_p50,
            min_speedup_p95=min_speedup_p95,
            reps=reps,
            warmup=warmup,
            thread_mode=thread_mode,
            thread_env=thread_env,
            cold_thread_mode=cold_thread_mode,
            cold_thread_env=cold_thread_env,
        )

    rust_p50 = _percentile(rust_times, 50)
    rust_p95 = _percentile(rust_times, 95)
    legacy_p50 = _percentile(legacy_times, 50)
    legacy_p95 = _percentile(legacy_times, 95)
    raw_passed, s50, s95 = _passes(
        legacy_p50,
        legacy_p95,
        rust_p50,
        rust_p95,
        min_speedup_p50=min_speedup_p50,
        min_speedup_p95=min_speedup_p95,
    )
    waivers = _perf_waivers_by_api_lane()
    waiver = waivers.get((api_id, lane)) or waivers.get((api_id, "*"), "")
    waived = bool(waiver and not raw_passed)
    passed = raw_passed or waived

    rust_cold = legacy_cold = speedup_cold = None
    legacy_cold_source = "not-measured"
    legacy_cold_cache_key = ""
    if measure_cold:
        try:
            rust_cold = _oracle.time_rust_cold(
                api_id, thread_mode=cold_thread_mode, **sample.rust_kwargs
            )
            legacy_cold, legacy_cold_source, legacy_cold_cache_key = _time_legacy_cold(
                api_id,
                sample.legacy_kwargs,
                seed=seed,
                cold_thread_mode=cold_thread_mode,
                lane=lane,
                workload_shape=workload_shape,
                workload_label=workload_label,
                legacy_cache=legacy_cache,
            )
            speedup_cold = legacy_cold / rust_cold if rust_cold > 0 else float("inf")
        except Exception as e:
            # Cold timing failure shouldn't fail the gate — record and move on.
            rust_cold = legacy_cold = speedup_cold = None
            print(f"  [cold-time error for {api_id} ({lane}): {e}]", file=sys.stderr)

    return SpeedResult(
        api_id=api_id,
        n=workload_rows,
        rust_p50=rust_p50,
        rust_p95=rust_p95,
        legacy_p50=legacy_p50,
        legacy_p95=legacy_p95,
        speedup_p50=s50,
        speedup_p95=s95,
        raw_passed=raw_passed,
        passed=passed,
        min_speedup_p50=min_speedup_p50,
        min_speedup_p95=min_speedup_p95,
        waived=waived,
        waiver=waiver,
        error=None,
        lane=lane,
        lane_description=lane_description,
        lane_enforced=lane_enforced,
        workload_shape=workload_shape,
        workload_label=workload_label,
        reps=reps,
        warmup=warmup,
        thread_mode=thread_mode,
        thread_env=thread_env,
        legacy_source=legacy_source,
        legacy_cache_key=legacy_cache_key,
        rust_cold=rust_cold,
        legacy_cold=legacy_cold,
        speedup_cold=speedup_cold,
        cold_thread_mode=cold_thread_mode,
        cold_thread_env=cold_thread_env,
        legacy_cold_source=legacy_cold_source,
        legacy_cold_cache_key=legacy_cold_cache_key,
    )


def build_speed_lanes(
    *,
    n: int,
    reps: int,
    warmup: int,
    measure_cold: bool,
    include_tiny: bool = False,
    tiny_reps: int | None = None,
    tiny_warmup: int | None = None,
    tiny_cold: bool = False,
    include_large: bool = False,
    large_n: int | None = None,
    large_reps: int | None = None,
    large_warmup: int | None = None,
    large_cold: bool = False,
    large_enforced: bool = True,
) -> list[SpeedLane]:
    from . import _inputs

    workloads = _inputs.lane_workloads(tiny_n=DEFAULT_TINY_N, small_n=n)
    lanes: list[SpeedLane] = []
    if include_tiny:
        lanes.append(
            SpeedLane(
                name=DEFAULT_TINY_LANE_NAME,
                description=(
                    f"One-off/small-call lane at default n={DEFAULT_TINY_N} "
                    "with API-specific overrides where a scalar API's true "
                    f"one-off shape is smaller; enforced for p50/p95 >= "
                    f"{DEFAULT_TINY_SPEEDUP:.1f}x."
                ),
                enforced=True,
                reps=tiny_reps if tiny_reps is not None else reps,
                warmup=tiny_warmup if tiny_warmup is not None else warmup,
                measure_cold=tiny_cold,
                min_speedup_p50=DEFAULT_TINY_SPEEDUP,
                min_speedup_p95=DEFAULT_TINY_SPEEDUP,
                api_workloads=workloads[DEFAULT_TINY_LANE_NAME],
            )
        )
    lanes.append(
        SpeedLane(
            name=DEFAULT_SMALL_LANE_NAME,
            description=(
                f"Historical promotion gate at default n={n} with "
                "API-specific overrides for expensive scalar optimizers; "
                f"enforced for p50/p95 >= {DEFAULT_SMALL_SPEEDUP:.1f}x pass/fail."
            ),
            enforced=True,
            reps=reps,
            warmup=warmup,
            measure_cold=measure_cold,
            min_speedup_p50=DEFAULT_SMALL_SPEEDUP,
            min_speedup_p95=DEFAULT_SMALL_SPEEDUP,
            api_workloads=workloads[DEFAULT_SMALL_LANE_NAME],
        )
    )
    if not include_large:
        return lanes

    if large_n is None:
        api_workloads = workloads[DEFAULT_LARGE_LANE_NAME]
        description = (
            "API-shaped large workload lane with structured axes; enforced "
            f"for p50/p95 >= {DEFAULT_LARGE_SPEEDUP:.1f}x."
        )
    else:
        api_workloads = {
            api_id: _inputs.WorkloadShape(large_n) for api_id in _inputs.all_api_ids()
        }
        description = (
            f"Ad-hoc large workload override at rows={large_n}; enforced "
            f"for p50/p95 >= {DEFAULT_LARGE_SPEEDUP:.1f}x. Canonical scripts "
            "use API-shaped workloads instead of this scalar override."
        )

    lanes.append(
        SpeedLane(
            name=DEFAULT_LARGE_LANE_NAME,
            description=description,
            enforced=large_enforced,
            reps=large_reps if large_reps is not None else reps,
            warmup=large_warmup if large_warmup is not None else warmup,
            measure_cold=large_cold,
            min_speedup_p50=DEFAULT_LARGE_SPEEDUP,
            min_speedup_p95=DEFAULT_LARGE_SPEEDUP,
            api_workloads=api_workloads,
        )
    )
    return lanes


def measure_all(
    api_ids: list[str],
    *,
    n: int,
    reps: int = 7,
    warmup: int = 1,
    seed: int = 20260425,
    measure_cold: bool = False,
    thread_mode: str = "single",
    cold_thread_mode: str = "multi-thread",
    legacy_cache: dict[str, object] | None = None,
) -> list[SpeedResult]:
    lanes = build_speed_lanes(
        n=n,
        reps=reps,
        warmup=warmup,
        measure_cold=measure_cold,
    )
    return measure_lanes(
        api_ids,
        lanes,
        seed=seed,
        thread_mode=thread_mode,
        cold_thread_mode=cold_thread_mode,
        legacy_cache=legacy_cache,
    )


def measure_lanes(
    api_ids: Sequence[str],
    lanes: Sequence[SpeedLane],
    *,
    seed: int = 20260425,
    thread_mode: str = "single",
    cold_thread_mode: str = "multi-thread",
    legacy_cache: dict[str, object] | None = None,
) -> list[SpeedResult]:
    results: list[SpeedResult] = []
    for lane in lanes:
        for api_id in api_ids:
            workload = lane.workload_for(api_id)
            results.append(
                measure(
                    api_id,
                    n=_shape_rows(workload),
                    reps=lane.reps,
                    warmup=lane.warmup,
                    seed=seed,
                    measure_cold=lane.measure_cold,
                    thread_mode=thread_mode,
                    cold_thread_mode=cold_thread_mode,
                    lane=lane.name,
                    lane_description=lane.description,
                    lane_enforced=lane.enforced,
                    workload=workload,
                    min_speedup_p50=lane.min_speedup_p50,
                    min_speedup_p95=lane.min_speedup_p95,
                    legacy_cache=legacy_cache,
                )
            )
    return results


def _result_mode(results: list[SpeedResult], attr: str, default: str) -> str:
    modes = {getattr(r, attr) for r in results}
    if not modes:
        return default
    if len(modes) == 1:
        return next(iter(modes))
    return "mixed"


def _result_env(results: list[SpeedResult], attr: str) -> dict[str, str | None]:
    snapshots = [getattr(r, attr) for r in results if getattr(r, attr)]
    if not snapshots:
        return _threading.snapshot_thread_env()
    first = snapshots[0]
    if all(snapshot == first for snapshot in snapshots):
        return first
    return {key: "<mixed>" for key in _threading.THREAD_ENV_KEYS}


def _lane_names(results: Sequence[SpeedResult]) -> list[str]:
    return list(dict.fromkeys(r.lane for r in results))


def _flag(r: SpeedResult) -> str:
    if r.error:
        flag = f"ERR ({r.error[:40]})"
    elif r.waived and not r.raw_passed:
        flag = f"WAIVED ({r.waiver})"
    elif r.passed:
        flag = "PASS"
    else:
        flag = "FAIL"
    if not r.lane_enforced:
        return f"DIAG {flag}"
    return flag


def format_summary(results: list[SpeedResult]) -> str:
    has_cold = any(r.rust_cold is not None for r in results)
    warm_mode = _result_mode(results, "thread_mode", "single")
    cold_mode = _result_mode(results, "cold_thread_mode", "multi-thread")
    lines = [
        f"Thread mode: warm={warm_mode}; cold={cold_mode if has_cold else 'not measured'}",
        "Note: single-thread caps thread pools only; SIMD/ILP remain enabled within each CPU core.",
    ]
    if has_cold:
        lines.append(
            f"{'Lane':11s}  {'API':50s}  {'n':>8s}  {'rust warm':>10s}  "
            f"{'leg warm':>10s}  {'×p50':>6s}  {'×p95':>6s}  "
            f"{'rust cold':>11s}  {'leg cold':>10s}  {'×cold':>6s}  flag"
        )
        lines.append("-" * 152)
    else:
        lines.append(
            f"{'Lane':11s}  {'API':50s}  {'n':>8s}  {'rust p50':>10s}  "
            f"{'leg p50':>10s}  {'×p50':>6s}  {'×p95':>6s}  flag"
        )
        lines.append("-" * 124)

    for r in results:
        flag = _flag(r)
        if has_cold:
            cold_str = (
                f"{r.rust_cold*1000:>10.1f}ms  {r.legacy_cold*1000:>9.1f}ms  "
                f"{r.speedup_cold:>5.2f}x"
                if r.rust_cold is not None
                else f"{'—':>11s}  {'—':>10s}  {'—':>6s}"
            )
            lines.append(
                f"{r.lane:11s}  {r.api_id:50s}  {r.n:>8d}  "
                f"{r.rust_p50*1e6:>9.1f}μs  {r.legacy_p50*1e6:>9.1f}μs  "
                f"{r.speedup_p50:>5.2f}x  {r.speedup_p95:>5.2f}x  "
                f"{cold_str}  {flag}"
            )
        else:
            lines.append(
                f"{r.lane:11s}  {r.api_id:50s}  {r.n:>8d}  "
                f"{r.rust_p50*1e6:>9.1f}μs  {r.legacy_p50*1e6:>9.1f}μs  "
                f"{r.speedup_p50:>5.2f}x  {r.speedup_p95:>5.2f}x  {flag}"
            )
    return "\n".join(lines)


def _lane_metadata(results: list[SpeedResult]) -> list[dict[str, object]]:
    lanes: list[dict[str, object]] = []
    for name in _lane_names(results):
        lane_results = [r for r in results if r.lane == name]
        n_values = sorted({r.n for r in lane_results})
        lane = lane_results[0]
        lanes.append(
            {
                "name": name,
                "description": lane.lane_description,
                "enforced": all(r.lane_enforced for r in lane_results),
                "reps": sorted({r.reps for r in lane_results}),
                "warmup": sorted({r.warmup for r in lane_results}),
                "measure_cold": any(r.rust_cold is not None for r in lane_results),
                "min_speedup_p50": lane.min_speedup_p50,
                "min_speedup_p95": lane.min_speedup_p95,
                "n": n_values[0] if len(n_values) == 1 else None,
                "n_values": n_values,
                "api_count": len(lane_results),
            }
        )
    return lanes


def _governance_passed(result: SpeedResult) -> bool:
    return result.passed or not result.lane_enforced


def _lane_status(results: list[SpeedResult]) -> dict[str, dict[str, object]]:
    status: dict[str, dict[str, object]] = {}
    for name in _lane_names(results):
        lane_results = [r for r in results if r.lane == name]
        status[name] = {
            "passed": all(_governance_passed(r) for r in lane_results),
            "raw_passed": all(r.raw_passed for r in lane_results),
            "enforced": all(r.lane_enforced for r in lane_results),
            "waived": [r.api_id for r in lane_results if r.waived],
            "failed": [r.api_id for r in lane_results if not _governance_passed(r)],
        }
    return status


def to_json(
    results: list[SpeedResult],
    legacy_cache: Mapping[str, object] | None = None,
) -> dict:
    artifact = {
        "default_small_speedup": DEFAULT_SMALL_SPEEDUP,
        "default_large_speedup": DEFAULT_LARGE_SPEEDUP,
        "default_tiny_speedup": DEFAULT_TINY_SPEEDUP,
        "thread_mode": _result_mode(results, "thread_mode", "single"),
        "thread_env": _result_env(results, "thread_env"),
        "cold_thread_mode": _result_mode(results, "cold_thread_mode", "multi-thread"),
        "cold_thread_env": _result_env(results, "cold_thread_env"),
        "thread_policy": (
            "Warm p50/p95 timings use thread_mode for apples-to-apples "
            "single-process/single-thread governance by default. Cold timings "
            "use cold_thread_mode and remain process-realistic by default. "
            "Thread caps do not disable SIMD or per-core instruction-level "
            "parallelism."
        ),
        "lane_policy": (
            "The tiny-n lane records quick one-off/small-call behavior. "
            "The small-n lane preserves the historical n=2000 promotion gate. "
            "The large-n lane is API-shaped, records structured workload axes, "
            "and is enforced by default with explicit per-lane waivers required "
            "for known large-workload regressions."
        ),
        "legacy_timing_cache": _legacy_cache_metadata(legacy_cache),
        "lanes": _lane_metadata(results),
        "lane_status": _lane_status(results),
        "apis": [
            {
                "api_id": r.api_id,
                "lane": r.lane,
                "lane_description": r.lane_description,
                "lane_enforced": r.lane_enforced,
                "workload_shape": r.workload_shape,
                "workload_label": r.workload_label,
                "n": r.n,
                "reps": r.reps,
                "warmup": r.warmup,
                "rust_p50_s": r.rust_p50,
                "rust_p95_s": r.rust_p95,
                "legacy_p50_s": r.legacy_p50,
                "legacy_p95_s": r.legacy_p95,
                "legacy_source": r.legacy_source,
                "legacy_cache_key": r.legacy_cache_key,
                "speedup_p50": r.speedup_p50,
                "speedup_p95": r.speedup_p95,
                "min_speedup_p50": r.min_speedup_p50,
                "min_speedup_p95": r.min_speedup_p95,
                "raw_passed": r.raw_passed,
                "passed": r.passed,
                "waived": r.waived,
                "waiver": r.waiver,
                "error": r.error,
                "rust_cold_s": r.rust_cold,
                "legacy_cold_s": r.legacy_cold,
                "legacy_cold_source": r.legacy_cold_source,
                "legacy_cold_cache_key": r.legacy_cold_cache_key,
                "speedup_cold": r.speedup_cold,
            }
            for r in results
        ],
        "all_passed": all(_governance_passed(r) for r in results),
    }
    if artifact["legacy_timing_cache"] is None:
        artifact.pop("legacy_timing_cache")
    return artifact


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rust-vs-baseline-main speedup gate.")
    p.add_argument(
        "--n",
        type=int,
        default=DEFAULT_SMALL_N,
        help="Workload size for the enforced small-n lane (default: 2000).",
    )
    p.add_argument(
        "--reps",
        type=int,
        default=7,
        help="Timing reps for the small-n lane (default: 7).",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup reps before timing the small-n lane (default: 1).",
    )
    p.add_argument(
        "--tiny",
        action="store_true",
        help="Also measure the enforced tiny-n one-off lane (default n=10).",
    )
    p.add_argument(
        "--tiny-reps",
        type=int,
        default=None,
        help="Timing reps for the tiny-n lane (default: same as --reps).",
    )
    p.add_argument(
        "--tiny-warmup",
        type=int,
        default=None,
        help="Warmup reps for the tiny-n lane (default: same as --warmup).",
    )
    p.add_argument(
        "--tiny-cold",
        action="store_true",
        help="Also collect cold-call timings for the tiny-n lane.",
    )
    p.add_argument(
        "--large",
        action="store_true",
        help="Also measure the enforced API-shaped large-n lane.",
    )
    p.add_argument(
        "--large-n",
        type=int,
        default=None,
        help="Override the large lane to use this n for every API.",
    )
    p.add_argument(
        "--large-reps",
        type=int,
        default=None,
        help="Timing reps for the large-n lane (default: same as --reps).",
    )
    p.add_argument(
        "--large-warmup",
        type=int,
        default=None,
        help="Warmup reps for the large-n lane (default: same as --warmup).",
    )
    p.add_argument(
        "--large-cold",
        action="store_true",
        help="Also collect cold-call timings for the large-n lane.",
    )
    p.add_argument(
        "--large-enforced",
        action="store_true",
        help="Deprecated compatibility flag; the large-n lane is enforced by default.",
    )
    p.add_argument(
        "--large-diagnostic",
        action="store_true",
        help="Ad-hoc escape hatch: measure large-n but exclude it from pass/fail.",
    )
    p.add_argument(
        "--apis",
        nargs="*",
        default=None,
        help="Specific API ids to gate. Defaults to ALL.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260425,
        help="RNG seed for the timing workload.",
    )
    p.add_argument(
        "--threads",
        choices=("single", "multi-thread", "native"),
        default="single",
        help=(
            "Warm p50/p95 thread policy (default: single). Use 'multi-thread' "
            "only for separately labeled scaling artifacts that allow both "
            "Rust Rayon and the legacy NumPy/JAX/BLAS pools to scale across "
            "available cores. 'native' is accepted as a deprecated alias "
            "for 'multi-thread'."
        ),
    )
    p.add_argument(
        "--cold-threads",
        choices=("single", "multi-thread", "native"),
        default="multi-thread",
        help=(
            "Cold-call thread policy when --cold is set (default: "
            "multi-thread for one-shot production realism). 'native' is "
            "accepted as a deprecated alias for 'multi-thread'."
        ),
    )
    p.add_argument(
        "--cold",
        action="store_true",
        help="Also measure cold-call latency for the small-n lane.",
    )
    p.add_argument(
        "--legacy-cache",
        type=Path,
        default=None,
        help=(
            "Optional JSON cache for baseline-main legacy warm/cold timings. "
            "When provided, missing or stale entries fail loudly unless "
            "--refresh-legacy-cache is also set."
        ),
    )
    p.add_argument(
        "--refresh-legacy-cache",
        action="store_true",
        help=(
            "Recapture missing/requested --legacy-cache entries and merge them "
            "into the existing cache. Existing entries for other lanes/APIs are "
            "preserved."
        ),
    )
    p.add_argument(
        "--replace-legacy-cache",
        action="store_true",
        help=(
            "With --refresh-legacy-cache, discard the existing cache before "
            "capturing requested lanes/APIs. Use after benchmark process or "
            "legacy identity changes."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON artifact.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    _threading.apply_thread_mode(args.threads)

    from . import _inputs

    if args.refresh_legacy_cache and args.legacy_cache is None:
        parser.error("--refresh-legacy-cache requires --legacy-cache")
    if args.replace_legacy_cache and not args.refresh_legacy_cache:
        parser.error("--replace-legacy-cache requires --refresh-legacy-cache")

    api_ids = args.apis or list(_inputs.all_api_ids())
    legacy_cache = prepare_legacy_timing_cache(
        args.legacy_cache,
        refresh=args.refresh_legacy_cache,
        replace=args.replace_legacy_cache,
    )
    lanes = build_speed_lanes(
        n=args.n,
        reps=args.reps,
        warmup=args.warmup,
        measure_cold=args.cold,
        include_tiny=args.tiny,
        tiny_reps=args.tiny_reps,
        tiny_warmup=args.tiny_warmup,
        tiny_cold=args.tiny_cold,
        include_large=args.large,
        large_n=args.large_n,
        large_reps=args.large_reps,
        large_warmup=args.large_warmup,
        large_cold=args.large_cold,
        large_enforced=args.large_enforced or not args.large_diagnostic,
    )
    results = measure_lanes(
        api_ids,
        lanes,
        seed=args.seed,
        thread_mode=args.threads,
        cold_thread_mode=args.cold_threads,
        legacy_cache=legacy_cache,
    )
    write_legacy_timing_cache(legacy_cache)
    artifact = to_json(results, legacy_cache=legacy_cache)
    print(format_summary(results))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(artifact, indent=2))
        print(f"\nwrote {args.output}", file=sys.stderr)

    return 0 if artifact["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
