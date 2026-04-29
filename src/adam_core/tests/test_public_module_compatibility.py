from __future__ import annotations

import ast
import importlib
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src" / "adam_core"
PUBLIC_COMPATIBILITY_DOC = (
    PROJECT_ROOT / "docs" / "source" / "reference" / "rust_public_compatibility.rst"
)

PUBLIC_MODULE_SYMBOLS = {
    "adam_core.dynamics.aberrations": (
        "_add_light_time",
        "_add_light_time_vmap",
        "add_light_time",
        "add_stellar_aberration",
    ),
    "adam_core.dynamics.barker": ("solve_barker",),
    "adam_core.dynamics.chi": ("ChiDiagnostics", "calc_chi", "calc_chi_diagnostics"),
    "adam_core.dynamics.kepler": (
        "calc_apoapsis_distance",
        "calc_mean_anomaly",
        "calc_mean_motion",
        "calc_periapsis_distance",
        "calc_period",
        "calc_semi_latus_rectum",
        "calc_semi_major_axis",
        "solve_kepler",
    ),
    "adam_core.dynamics.lagrange": (
        "apply_lagrange_coefficients",
        "calc_lagrange_coefficients",
    ),
    "adam_core.dynamics.stumpff": ("calc_stumpff",),
    "adam_core.coordinates.jacobian": ("calc_jacobian",),
}

RESTORED_COMPATIBILITY_FILES = {
    SRC_ROOT / "dynamics" / "aberrations.py",
    SRC_ROOT / "dynamics" / "barker.py",
    SRC_ROOT / "dynamics" / "chi.py",
    SRC_ROOT / "dynamics" / "kepler.py",
    SRC_ROOT / "dynamics" / "lagrange.py",
    SRC_ROOT / "dynamics" / "stumpff.py",
    SRC_ROOT / "coordinates" / "jacobian.py",
}


def _module_name_for_path(path: Path) -> str:
    relative = path.relative_to(PROJECT_ROOT / "src").with_suffix("")
    return ".".join(relative.parts)


def _resolve_import_from(module_name: str, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module

    package_parts = module_name.split(".")[:-1]
    if node.level > len(package_parts) + 1:
        return None

    base_parts = package_parts[: len(package_parts) - node.level + 1]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


def _is_test_path(path: Path) -> bool:
    return "tests" in path.relative_to(SRC_ROOT).parts


@pytest.mark.parametrize("module_name", sorted(PUBLIC_MODULE_SYMBOLS))
def test_baseline_public_module_imports(module_name: str) -> None:
    module = importlib.import_module(module_name)
    for symbol in PUBLIC_MODULE_SYMBOLS[module_name]:
        assert hasattr(module, symbol), f"{module_name} is missing {symbol}"


def test_public_surface_inventory_documents_every_restored_symbol() -> None:
    doc = PUBLIC_COMPATIBILITY_DOC.read_text()
    for module_name, symbols in PUBLIC_MODULE_SYMBOLS.items():
        for symbol in symbols:
            assert f"``{module_name}.{symbol}``" in doc


def test_production_code_does_not_import_restored_compatibility_modules() -> None:
    forbidden = set(PUBLIC_MODULE_SYMBOLS)
    offenders: list[str] = []

    for path in SRC_ROOT.rglob("*.py"):
        if path in RESTORED_COMPATIBILITY_FILES or _is_test_path(path):
            continue

        module_name = _module_name_for_path(path)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported = alias.name
                    if any(
                        imported == forbidden_module
                        or imported.startswith(f"{forbidden_module}.")
                        for forbidden_module in forbidden
                    ):
                        offenders.append(f"{path}: import {imported}")

            if isinstance(node, ast.ImportFrom):
                imported = _resolve_import_from(module_name, node)
                if imported in forbidden:
                    offenders.append(f"{path}: from {imported} import ...")

    assert offenders == []


def test_kepler_compatibility_smoke() -> None:
    kepler = importlib.import_module("adam_core.dynamics.kepler")

    np.testing.assert_allclose(kepler.calc_periapsis_distance(2.0, 0.25), 1.5)
    np.testing.assert_allclose(kepler.calc_apoapsis_distance(2.0, 0.25), 2.5)
    np.testing.assert_allclose(kepler.calc_semi_major_axis(1.5, 0.25), 2.0)
    np.testing.assert_allclose(kepler.calc_semi_latus_rectum(2.0, 0.25), 1.875)
    assert np.isfinite(np.asarray(kepler.solve_kepler(0.1, 0.2)))


def test_universal_variable_compatibility_smoke() -> None:
    stumpff = importlib.import_module("adam_core.dynamics.stumpff")
    chi = importlib.import_module("adam_core.dynamics.chi")
    lagrange = importlib.import_module("adam_core.dynamics.lagrange")

    c0, c1, c2, c3, c4, c5 = stumpff.calc_stumpff(0.0)
    np.testing.assert_allclose(
        np.asarray([c0, c1, c2, c3, c4, c5]), [1, 1, 0.5, 1 / 6, 1 / 24, 1 / 120]
    )

    r = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v = np.array([0.0, 0.017, 0.0], dtype=np.float64)
    chi_out = chi.calc_chi(r, v, 1.0)
    assert np.isfinite(np.asarray(chi_out[0]))

    coeffs, _, _ = lagrange.calc_lagrange_coefficients(r, v, 1.0)
    r_new, v_new = lagrange.apply_lagrange_coefficients(r, v, *coeffs)
    assert np.isfinite(np.asarray(r_new)).all()
    assert np.isfinite(np.asarray(v_new)).all()


def test_aberrations_compatibility_smoke() -> None:
    aberrations = importlib.import_module("adam_core.dynamics.aberrations")

    orbit = np.array([[1.0, 0.0, 0.0, 0.0, 0.017, 0.0]], dtype=np.float64)
    observer_position = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    corrected, light_time = aberrations.add_light_time(
        orbit,
        np.array([60000.0], dtype=np.float64),
        observer_position,
    )
    assert corrected.shape == (1, 6)
    assert light_time.shape == (1,)
    assert np.isfinite(corrected).all()
    assert np.isfinite(light_time).all()

    observer_state = np.zeros((1, 6), dtype=np.float64)
    aberrated = aberrations.add_stellar_aberration(orbit, observer_state)
    np.testing.assert_allclose(aberrated, orbit[:, :3])


def test_jacobian_compatibility_smoke() -> None:
    jacobian = importlib.import_module("adam_core.coordinates.jacobian")

    def square(coords: np.ndarray) -> np.ndarray:
        return coords * coords

    result = jacobian.calc_jacobian(
        np.array([[2.0, 3.0]], dtype=np.float64),
        square,
    )
    assert result.shape == (1, 2, 2)
    np.testing.assert_allclose(result[0], np.diag([4.0, 6.0]))
