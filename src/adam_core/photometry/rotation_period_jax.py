from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from math import ceil

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

jax.config.update("jax_enable_x64", True)


@dataclass(slots=True)
class JAXBatchFitResult:
    scores: npt.NDArray[np.float64]
    best_valid: bool
    best_coeffs: npt.NDArray[np.float64]
    best_mask: npt.NDArray[np.bool_]
    best_sigma: float
    best_rss: float
    best_df: int
    best_n_fit: int
    best_n_clipped: int


def _next_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return int(value)
    return int(ceil(value / multiple) * multiple)


def _pad_rows(
    *,
    time_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    fixed: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None,
    row_pad_multiple: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
]:
    n_rows = int(y.shape[0])
    n_padded = _next_multiple(n_rows, row_pad_multiple)
    time_pad = np.zeros(n_padded, dtype=np.float64)
    y_pad = np.zeros(n_padded, dtype=np.float64)
    fixed_pad = np.zeros((n_padded, fixed.shape[1]), dtype=np.float64)
    weights_pad = np.zeros(n_padded, dtype=np.float64)
    row_mask = np.zeros(n_padded, dtype=bool)

    time_pad[:n_rows] = time_rel
    y_pad[:n_rows] = y
    fixed_pad[:n_rows, :] = fixed
    weights_pad[:n_rows] = (
        np.ones(n_rows, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    row_mask[:n_rows] = True
    return time_pad, y_pad, fixed_pad, weights_pad, row_mask


@partial(jax.jit, static_argnames=("fourier_order",))
def _build_fourier_batch(
    time_rel: jnp.ndarray,
    frequencies: jnp.ndarray,
    *,
    fourier_order: int,
) -> jnp.ndarray:
    phase = 2.0 * jnp.pi * frequencies[:, None] * time_rel[None, :]
    cols = []
    for harmonic in range(1, fourier_order + 1):
        harmonic_phase = harmonic * phase
        cols.append(jnp.cos(harmonic_phase))
        cols.append(jnp.sin(harmonic_phase))
    return jnp.stack(cols, axis=2)


@partial(jax.jit, static_argnames=("fourier_order", "max_clip_iterations"))
def _evaluate_frequency_batch_jit(
    time_rel: jnp.ndarray,
    y: jnp.ndarray,
    fixed: jnp.ndarray,
    weights: jnp.ndarray,
    row_mask: jnp.ndarray,
    frequencies: jnp.ndarray,
    frequency_valid: jnp.ndarray,
    *,
    fourier_order: int,
    clip_sigma: float,
    max_clip_iterations: int,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
    fourier = _build_fourier_batch(
        time_rel,
        frequencies,
        fourier_order=fourier_order,
    )
    design = jnp.concatenate(
        [
            jnp.broadcast_to(
                fixed[None, :, :],
                (frequencies.shape[0], fixed.shape[0], fixed.shape[1]),
            ),
            fourier,
        ],
        axis=2,
    )
    n_par = design.shape[2]
    eye = jnp.eye(n_par, dtype=jnp.float64)
    target = jnp.broadcast_to(y[None, :], (frequencies.shape[0], y.shape[0]))
    active = jnp.broadcast_to(row_mask[None, :], target.shape) & frequency_valid[:, None]

    def solve_with_mask(active_mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        w_eff = jnp.where(active_mask, weights[None, :], 0.0)
        sqrt_w = jnp.sqrt(w_eff)
        design_w = design * sqrt_w[:, :, None]
        target_w = target * sqrt_w
        xt = jnp.swapaxes(design_w, 1, 2)
        xtx = jnp.matmul(xt, design_w)
        xty = jnp.matmul(xt, target_w[:, :, None])[:, :, 0]
        coeffs = jnp.linalg.solve(
            xtx + 1.0e-12 * eye[None, :, :],
            xty[:, :, None],
        )[:, :, 0]
        model = jnp.matmul(design, coeffs[:, :, None])[:, :, 0]
        resid = target - model
        rss = jnp.sum(jnp.where(active_mask, w_eff * resid * resid, 0.0), axis=1)
        n_fit = jnp.sum(active_mask, axis=1)
        df = n_fit - n_par
        sigma = jnp.sqrt(jnp.where(df > 0, rss / df, jnp.inf))
        return coeffs, resid, rss, sigma

    coeffs = jnp.zeros((frequencies.shape[0], n_par), dtype=jnp.float64)
    resid = jnp.zeros_like(target)
    rss = jnp.full((frequencies.shape[0],), jnp.inf, dtype=jnp.float64)
    sigma = jnp.full((frequencies.shape[0],), jnp.inf, dtype=jnp.float64)

    for _ in range(max_clip_iterations):
        coeffs, resid, rss, sigma = solve_with_mask(active)
        clip_limit = clip_sigma * sigma[:, None]
        keep = jnp.abs(resid) <= clip_limit
        new_active = active & keep
        active = jnp.where(
            frequency_valid[:, None],
            new_active,
            active,
        )

    coeffs, resid, rss, sigma = solve_with_mask(active)
    n_fit = jnp.sum(active, axis=1)
    df = n_fit - n_par
    valid = frequency_valid & (df > 0) & jnp.isfinite(sigma)
    scores = jnp.where(valid, sigma, jnp.inf)
    best_idx = jnp.argmin(scores)
    return (
        scores,
        valid,
        coeffs[best_idx],
        active[best_idx],
        sigma[best_idx],
        rss[best_idx],
        df[best_idx],
        n_fit[best_idx],
    )


def evaluate_frequency_indices_jax(
    *,
    time_rel: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    fixed: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None,
    frequencies: npt.NDArray[np.float64],
    sample_indices: npt.NDArray[np.int64],
    fourier_order: int,
    clip_sigma: float,
    jax_batch_size: int,
    row_pad_multiple: int,
    max_clip_iterations: int,
) -> JAXBatchFitResult:
    if sample_indices.size == 0:
        return JAXBatchFitResult(
            scores=np.zeros(0, dtype=np.float64),
            best_valid=False,
            best_coeffs=np.zeros(fixed.shape[1] + 2 * fourier_order, dtype=np.float64),
            best_mask=np.zeros(time_rel.shape[0], dtype=bool),
            best_sigma=float("inf"),
            best_rss=float("inf"),
            best_df=0,
            best_n_fit=0,
            best_n_clipped=int(time_rel.shape[0]),
        )

    time_pad, y_pad, fixed_pad, weights_pad, row_mask = _pad_rows(
        time_rel=np.asarray(time_rel, dtype=np.float64),
        y=np.asarray(y, dtype=np.float64),
        fixed=np.asarray(fixed, dtype=np.float64),
        weights=None if weights is None else np.asarray(weights, dtype=np.float64),
        row_pad_multiple=row_pad_multiple,
    )
    n_scores = int(sample_indices.size)
    scores = np.full(n_scores, np.nan, dtype=np.float64)

    best_valid = False
    best_sigma = float("inf")
    best_coeffs = np.zeros(fixed.shape[1] + 2 * fourier_order, dtype=np.float64)
    best_mask = np.zeros(time_rel.shape[0], dtype=bool)
    best_rss = float("inf")
    best_df = 0
    best_n_fit = 0

    batch_size = max(1, int(jax_batch_size))
    n_batches = int(ceil(n_scores / batch_size))
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        stop = min(start + batch_size, n_scores)
        batch_indices = np.asarray(sample_indices[start:stop], dtype=np.int64)
        frequencies_batch = np.zeros(batch_size, dtype=np.float64)
        frequency_valid = np.zeros(batch_size, dtype=bool)
        valid_count = stop - start
        frequencies_batch[:valid_count] = frequencies[batch_indices]
        frequency_valid[:valid_count] = True

        (
            scores_batch,
            valid_batch,
            coeffs_batch,
            mask_batch,
            sigma_batch,
            rss_batch,
            df_batch,
            n_fit_batch,
        ) = _evaluate_frequency_batch_jit(
            jnp.asarray(time_pad),
            jnp.asarray(y_pad),
            jnp.asarray(fixed_pad),
            jnp.asarray(weights_pad),
            jnp.asarray(row_mask),
            jnp.asarray(frequencies_batch),
            jnp.asarray(frequency_valid),
            fourier_order=int(fourier_order),
            clip_sigma=float(clip_sigma),
            max_clip_iterations=int(max_clip_iterations),
        )

        scores_np = np.asarray(scores_batch, dtype=np.float64)[:valid_count]
        valid_np = np.asarray(valid_batch, dtype=bool)[:valid_count]
        scores[start:stop] = np.where(valid_np, scores_np, np.nan)

        if np.any(valid_np):
            local_idx = int(np.nanargmin(np.where(valid_np, scores_np, np.nan)))
            local_sigma = float(scores_np[local_idx])
            if local_sigma < best_sigma:
                best_valid = True
                best_sigma = local_sigma
                best_coeffs = np.asarray(coeffs_batch, dtype=np.float64)
                best_mask = np.asarray(mask_batch, dtype=bool)[: time_rel.shape[0]]
                best_rss = float(np.asarray(rss_batch, dtype=np.float64))
                best_df = int(np.asarray(df_batch, dtype=np.int64))
                best_n_fit = int(np.asarray(n_fit_batch, dtype=np.int64))

    return JAXBatchFitResult(
        scores=scores,
        best_valid=best_valid,
        best_coeffs=best_coeffs,
        best_mask=best_mask,
        best_sigma=best_sigma,
        best_rss=best_rss,
        best_df=best_df,
        best_n_fit=best_n_fit,
        best_n_clipped=int(time_rel.shape[0] - best_n_fit),
    )
