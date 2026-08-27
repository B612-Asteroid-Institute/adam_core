# Waiver Schema and Review Rules

Last updated: 2026-04-14

## Purpose

Waivers allow legacy/JAX fallback to remain for a specific API when Rust parity or performance requirements are not yet met.

## Required Waiver Fields

Each waiver entry in `migration/waivers.yaml` must include:

- `id`: unique waiver id
- `api`: migrated API id (must match an entry in `src/adam_core/_rust/status.py`)
- `owner`: responsible engineer/team
- `reason`: concise technical reason fallback is retained
- `created`: date created (YYYY-MM-DD)
- `review_by`: next mandatory review date (YYYY-MM-DD)
- `exit_criteria`: objective condition to remove waiver
- `status`: `active` or `closed`

## Example Entry

```yaml
waivers:
  - id: waiver-001
    api: dynamics.calc_mean_motion
    owner: dynamics-team
    reason: rust p95 regression on high-eccentricity benchmark set
    created: 2026-04-14
    review_by: 2026-04-28
    exit_criteria: p50 and p95 speedup >= 1.2x with parity tests passing
    status: active
```

## Review Policy

- Review every active waiver at each milestone boundary.
- Close waiver as soon as exit criteria are met.
- Do not ship new long-lived fallback paths without a waiver record.
