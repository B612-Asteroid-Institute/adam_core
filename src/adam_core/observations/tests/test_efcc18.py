"""Tests for the EFCC18 star-catalog debiasing module (synthetic tables only)."""

from __future__ import annotations

import io
import os
import sys
import tarfile
import types
from pathlib import Path
from typing import Any

import healpy as hp
import numpy as np
import numpy.typing as npt
import pytest

from ..efcc18 import (
    EFCC18_ARCHIVE_URL,
    EFCC18_BIAS_DAT_ENV,
    EFCC18_BIAS_DAT_SHA256,
    EFCC18_BIAS_VERSION,
    EFCC18_CACHE_DIR_ENV,
    EFCC18_CATALOG_CODES,
    EFCC18_JPL_UNDEBIASED_ASTCATS,
    EFCC18_N_CATALOGS,
    EFCC18_N_TILES,
    EFCC18_NSIDE,
    MPC_ASTCAT_TO_EFCC18,
    compute_efcc18_corrections,
    download_efcc18_bias_table,
    efcc18_cache_dir,
    install_efcc18_bias_table,
    is_efcc18_covered,
    load_efcc18_biases,
    n_observations_covered,
    ra_dec_to_healpix,
    read_efcc18_bias_version,
    resolve_bias_dat,
)

JD_J2000 = 2451545.0
UCAC4_COLUMN = EFCC18_CATALOG_CODES.index("q")


def synthetic_bias_values() -> npt.NDArray[np.float32]:
    """Deterministic (tile, catalog, component)-encoded table."""
    tiles = np.arange(EFCC18_N_TILES, dtype=np.float32)[:, None]
    cats = np.arange(EFCC18_N_CATALOGS, dtype=np.float32)[None, :]
    table = np.zeros((EFCC18_N_TILES, EFCC18_N_CATALOGS, 4), dtype=np.float32)
    table[:, :, 0] = 0.001 * cats + 0.0001 * np.mod(tiles, 7)  # dRA arcsec
    table[:, :, 1] = 0.002 * cats + 0.0001 * np.mod(tiles, 5)  # dDec arcsec
    table[:, :, 2] = 0.0  # pmRA mas/yr
    table[:, :, 3] = 0.0  # pmDec mas/yr
    return table


def write_synthetic_bias_dat(path: Path, table: npt.NDArray[np.float32]) -> Path:
    """Write ``table`` in the bias.dat text layout (header + blank line)."""
    header = (
        "! BIAS_VERSION= synthetic (unit test)\n"
        f"! NSIDE= {EFCC18_NSIDE}\n"
        f"! NPIX= {EFCC18_N_TILES}\n"
        "!\n"
    )
    with path.open("w") as f:
        f.write(header)
        f.write("\n")  # blank line: the real file has these; loader must skip
        np.savetxt(f, table.reshape(EFCC18_N_TILES, EFCC18_N_CATALOGS * 4), fmt="%.4f")
    return path


@pytest.fixture(scope="module")
def synthetic_table() -> npt.NDArray[np.float32]:
    return synthetic_bias_values()


@pytest.fixture(scope="module")
def synthetic_bias_dat(
    tmp_path_factory: pytest.TempPathFactory, synthetic_table: npt.NDArray[np.float32]
) -> Path:
    directory = tmp_path_factory.mktemp("efcc18")
    return write_synthetic_bias_dat(directory / "bias.dat", synthetic_table)


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """
    Point the cache at an empty directory, clear the bias.dat override and hide
    any installed jpl_debias_2018 package (a None entry makes import fail).
    """
    cache = tmp_path / "cache"
    monkeypatch.setenv(EFCC18_CACHE_DIR_ENV, str(cache))
    monkeypatch.delenv(EFCC18_BIAS_DAT_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "jpl_debias_2018", None)
    return cache


def fake_data_package(monkeypatch: pytest.MonkeyPatch, bias_dat: Path) -> None:
    """Install a stand-in for the jpl_debias_2018 data package."""
    module = types.ModuleType("jpl_debias_2018")
    module.bias_dat = str(bias_dat)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "jpl_debias_2018", module)


class TestConstants:
    def test_catalog_codes_unique_and_sized(self) -> None:
        assert len(EFCC18_CATALOG_CODES) == 26 == EFCC18_N_CATALOGS
        assert len(set(EFCC18_CATALOG_CODES)) == 26
        assert "q" in EFCC18_CATALOG_CODES  # UCAC-4
        assert "U" in EFCC18_CATALOG_CODES  # Gaia-DR1
        assert "Y" in EFCC18_CATALOG_CODES  # UCAC-5 (relabelled from W in 2023)
        assert EFCC18_N_TILES == 12 * 64 * 64

    def test_mpc_astcat_map_targets_efcc18_columns_only(self) -> None:
        for mpc_name, code in MPC_ASTCAT_TO_EFCC18.items():
            assert code in EFCC18_CATALOG_CODES, (mpc_name, code)
        # Every EFCC18 column is reachable from exactly one MPC code.
        assert sorted(MPC_ASTCAT_TO_EFCC18.values()) == sorted(EFCC18_CATALOG_CODES)
        # Catalogs that postdate EFCC18 must not be mapped.
        for absent in ("Gaia2", "Gaia3", "Gaia3E", "ATLAS", "ATLAS2", "PS1_DR1"):
            assert absent not in MPC_ASTCAT_TO_EFCC18

    def test_jpl_undebiased_catalogs_are_tabulated(self) -> None:
        for astcat in EFCC18_JPL_UNDEBIASED_ASTCATS:
            assert astcat in MPC_ASTCAT_TO_EFCC18


class TestHealpix:
    def test_basic(self) -> None:
        tiles = ra_dec_to_healpix([0.0, 180.0], [0.0, 0.0])
        assert tiles.shape == (2,)
        assert tiles.dtype == np.int64
        assert tiles[0] != tiles[1]
        assert np.all((tiles >= 0) & (tiles < EFCC18_N_TILES))

    def test_tile_centers_round_trip(self) -> None:
        # pix2ang(nest) centers must map back onto their own tile index.
        rng = np.random.default_rng(42)
        tiles = np.concatenate(
            [[0, 1, EFCC18_N_TILES - 1], rng.integers(0, EFCC18_N_TILES, 200)]
        )
        theta, phi = hp.pix2ang(EFCC18_NSIDE, tiles, nest=True)
        dec = 90.0 - np.rad2deg(theta)
        ra = np.rad2deg(phi)
        np.testing.assert_array_equal(ra_dec_to_healpix(ra, dec), tiles)

    def test_ra_wrap_is_harmless(self) -> None:
        assert (
            ra_dec_to_healpix([-10.0], [5.0])[0] == ra_dec_to_healpix([350.0], [5.0])[0]
        )
        assert (
            ra_dec_to_healpix([370.0], [5.0])[0] == ra_dec_to_healpix([10.0], [5.0])[0]
        )


class TestLoader:
    def test_round_trip_and_cache(
        self, synthetic_bias_dat: Path, synthetic_table: npt.NDArray[np.float32]
    ) -> None:
        cache = synthetic_bias_dat.with_suffix(".npy")
        if cache.exists():
            cache.unlink()
        table = load_efcc18_biases(synthetic_bias_dat)
        assert table.shape == (EFCC18_N_TILES, EFCC18_N_CATALOGS, 4)
        assert table.dtype == np.float32
        np.testing.assert_allclose(table, synthetic_table, atol=5e-5)
        # Known cell: tile 10, catalog 'g' (index 5)
        np.testing.assert_allclose(
            table[10, 5, 0], 0.001 * 5 + 0.0001 * (10 % 7), atol=5e-5
        )
        assert cache.exists()
        cached = load_efcc18_biases(synthetic_bias_dat)
        np.testing.assert_array_equal(table, cached)

    def test_stale_or_malformed_cache_is_reparsed(
        self, synthetic_bias_dat: Path, tmp_path: Path
    ) -> None:
        bogus = tmp_path / "bogus.npy"
        np.save(bogus, np.zeros((3, 3)))
        # make sure the bogus cache is newer than bias.dat
        os.utime(bogus, None)
        table = load_efcc18_biases(synthetic_bias_dat, cache_npy=bogus)
        assert table.shape == (EFCC18_N_TILES, EFCC18_N_CATALOGS, 4)
        assert np.load(bogus).shape == (EFCC18_N_TILES, EFCC18_N_CATALOGS, 4)

    def test_bad_layout_rejected(self, tmp_path: Path) -> None:
        short = tmp_path / "bias.dat"
        with short.open("w") as f:
            f.write("! BIAS_VERSION= bad\n")
            np.savetxt(f, np.zeros((10, EFCC18_N_CATALOGS * 4)), fmt="%.4f")
        with pytest.raises(ValueError, match="Unexpected bias.dat layout"):
            load_efcc18_biases(short)

    def test_read_bias_version(self, synthetic_bias_dat: Path, tmp_path: Path) -> None:
        assert read_efcc18_bias_version(synthetic_bias_dat) == "synthetic (unit test)"
        no_header = tmp_path / "plain.dat"
        no_header.write_text("0.0 0.0\n")
        assert read_efcc18_bias_version(no_header) is None


class TestCorrections:
    def test_known_catalog_reads_the_right_cell(
        self, synthetic_table: npt.NDArray[np.float32]
    ) -> None:
        ra, dec = np.array([100.0]), np.array([30.0])
        out = compute_efcc18_corrections(
            ra, dec, ["UCAC4"], [JD_J2000], bias_table=synthetic_table
        )
        assert out.shape == (1, 2) and out.dtype == np.float64
        tile = int(ra_dec_to_healpix(ra, dec)[0])
        np.testing.assert_allclose(out[0, 0], synthetic_table[tile, UCAC4_COLUMN, 0])
        np.testing.assert_allclose(out[0, 1], synthetic_table[tile, UCAC4_COLUMN, 1])

    def test_uncovered_and_unknown_are_zero(
        self, synthetic_table: npt.NDArray[np.float32]
    ) -> None:
        out = compute_efcc18_corrections(
            [10.0, 200.0, 40.0],
            [-5.0, 25.0, 0.0],
            ["Gaia2", "Gaia3E", None],
            [2459200.0, 2460000.0, 2460000.0],
            bias_table=synthetic_table,
        )
        np.testing.assert_array_equal(out, np.zeros((3, 2)))

    def test_proper_motion_term(self) -> None:
        table = np.zeros((EFCC18_N_TILES, EFCC18_N_CATALOGS, 4), dtype=np.float32)
        theta, phi = hp.pix2ang(EFCC18_NSIDE, 0, nest=True)
        ra, dec = float(np.rad2deg(phi)), float(90.0 - np.rad2deg(theta))
        table[0, UCAC4_COLUMN] = [0.0, 0.0, 100.0, -50.0]  # mas/yr
        ten_years = JD_J2000 + 10.0 * 365.25
        out = compute_efcc18_corrections(
            [ra], [dec], ["UCAC4"], [ten_years], bias_table=table
        )
        np.testing.assert_allclose(out[0], [1.0, -0.5], atol=1e-9)
        # Linear in time: 20 years -> double
        out20 = compute_efcc18_corrections(
            [ra], [dec], ["UCAC4"], [JD_J2000 + 20.0 * 365.25], bias_table=table
        )
        np.testing.assert_allclose(out20[0], [2.0, -1.0], atol=1e-9)

    def test_exclude_astcats(self, synthetic_table: npt.NDArray[np.float32]) -> None:
        args = ([10.0, 10.0], [5.0, 5.0], ["Tycho", "UCAC4"], [JD_J2000, JD_J2000])
        both = compute_efcc18_corrections(*args, bias_table=synthetic_table)
        assert np.all(both != 0.0)
        excl = compute_efcc18_corrections(
            *args,
            bias_table=synthetic_table,
            exclude_astcats=EFCC18_JPL_UNDEBIASED_ASTCATS,
        )
        np.testing.assert_array_equal(excl[0], [0.0, 0.0])
        np.testing.assert_array_equal(excl[1], both[1])

    def test_validation(self, synthetic_table: npt.NDArray[np.float32]) -> None:
        with pytest.raises(ValueError, match="length N"):
            compute_efcc18_corrections(
                [1.0, 2.0],
                [0.0],
                ["UCAC4", "UCAC4"],
                [JD_J2000, JD_J2000],
                bias_table=synthetic_table,
            )
        with pytest.raises(ValueError, match="bias_table has shape"):
            compute_efcc18_corrections(
                [1.0], [0.0], ["UCAC4"], [JD_J2000], bias_table=np.zeros((3, 26, 4))
            )
        # No covered rows: no table needed at all (must not try to load one)
        out = compute_efcc18_corrections(
            [1.0], [0.0], ["Gaia2"], [JD_J2000], bias_table=np.zeros((3, 26, 4))
        )
        np.testing.assert_array_equal(out, np.zeros((1, 2)))


class TestCoverage:
    def test_is_covered_and_count(self) -> None:
        astcats = ["UCAC4", "Gaia2", "Gaia3", "USNOA2", None, "Tycho"]
        np.testing.assert_array_equal(
            is_efcc18_covered(astcats), [True, False, False, True, False, True]
        )
        assert n_observations_covered(astcats) == 3
        assert n_observations_covered(astcats, exclude_astcats=("Tycho",)) == 2
        assert n_observations_covered(["Gaia2", None]) == 0
        assert n_observations_covered([]) == 0


class TestLocateAndInstall:
    def test_cache_dir_resolution(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv(EFCC18_CACHE_DIR_ENV, str(tmp_path / "explicit"))
        assert efcc18_cache_dir() == tmp_path / "explicit"
        monkeypatch.delenv(EFCC18_CACHE_DIR_ENV)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        assert efcc18_cache_dir() == tmp_path / "xdg" / "adam_core" / "efcc18"
        monkeypatch.delenv("XDG_CACHE_HOME")
        assert efcc18_cache_dir() == Path.home() / ".cache" / "adam_core" / "efcc18"

    def test_resolve_order(
        self,
        isolated_env: Path,
        synthetic_bias_dat: Path,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Nothing anywhere: helpful error naming the data package
        with pytest.raises(FileNotFoundError, match="jpl-debias-2018"):
            resolve_bias_dat()
        with pytest.raises(FileNotFoundError, match="not found at"):
            resolve_bias_dat(tmp_path / "missing.dat")
        # Cache directory copy
        isolated_env.mkdir(parents=True)
        cached = isolated_env / "bias.dat"
        cached.write_text(synthetic_bias_dat.read_text())
        assert resolve_bias_dat() == cached
        # Environment variable wins over the cache
        monkeypatch.setenv(EFCC18_BIAS_DAT_ENV, str(synthetic_bias_dat))
        assert resolve_bias_dat() == synthetic_bias_dat
        monkeypatch.setenv(EFCC18_BIAS_DAT_ENV, str(tmp_path / "nope.dat"))
        with pytest.raises(FileNotFoundError, match=EFCC18_BIAS_DAT_ENV):
            resolve_bias_dat()
        # Explicit path wins over everything
        assert resolve_bias_dat(cached) == cached

    def test_resolve_uses_installed_data_package(
        self,
        isolated_env: Path,
        synthetic_bias_dat: Path,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Package beats the cache directory...
        isolated_env.mkdir(parents=True)
        cached = isolated_env / "bias.dat"
        cached.write_text("cache copy\n")
        fake_data_package(monkeypatch, synthetic_bias_dat)
        assert resolve_bias_dat() == synthetic_bias_dat
        # ...the environment variable beats the package...
        monkeypatch.setenv(EFCC18_BIAS_DAT_ENV, str(cached))
        assert resolve_bias_dat() == cached
        monkeypatch.delenv(EFCC18_BIAS_DAT_ENV)
        # ...and a package whose file is missing falls through to the cache.
        fake_data_package(monkeypatch, tmp_path / "gone" / "bias.dat")
        assert resolve_bias_dat() == cached
        # A package object without the attribute is ignored too.
        monkeypatch.setitem(
            sys.modules, "jpl_debias_2018", types.ModuleType("jpl_debias_2018")
        )
        assert resolve_bias_dat() == cached

    def test_cache_falls_back_to_cache_dir_when_source_dir_is_read_only(
        self,
        isolated_env: Path,
        synthetic_bias_dat: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from .. import efcc18 as efcc18_module

        sibling = synthetic_bias_dat.with_suffix(".npy")
        if sibling.exists():
            sibling.unlink()
        monkeypatch.setattr(efcc18_module, "_is_writable", lambda directory: False)
        table = load_efcc18_biases(synthetic_bias_dat)
        assert table.shape == (EFCC18_N_TILES, EFCC18_N_CATALOGS, 4)
        assert not sibling.exists()
        fallback = isolated_env / "bias.npy"
        assert fallback.exists()
        # Second load comes from the fallback cache
        np.testing.assert_array_equal(load_efcc18_biases(synthetic_bias_dat), table)

    def test_install_from_file_and_archive(
        self, isolated_env: Path, synthetic_bias_dat: Path, tmp_path: Path
    ) -> None:
        # Synthetic content does not match the published checksum...
        with pytest.raises(ValueError, match="SHA-256"):
            install_efcc18_bias_table(synthetic_bias_dat)
        assert not (isolated_env / "bias.dat").exists()  # nothing left behind
        # ...so opt out of verification for the synthetic table.
        installed = install_efcc18_bias_table(synthetic_bias_dat, verify_checksum=False)
        assert installed == isolated_env / "bias.dat"
        assert installed.read_bytes() == synthetic_bias_dat.read_bytes()
        assert resolve_bias_dat() == installed

        # A stale parsed cache next to the install is removed.
        stale = installed.with_suffix(".npy")
        np.save(stale, np.zeros((2, 2)))
        archive = tmp_path / "debias_2018.tgz"
        with tarfile.open(archive, "w:gz") as tar:
            readme = tmp_path / "README.txt"
            readme.write_text("synthetic archive\n")
            tar.add(readme, arcname="README.txt")
            tar.add(synthetic_bias_dat, arcname="bias.dat")
        installed = install_efcc18_bias_table(archive, verify_checksum=False)
        assert installed.read_bytes() == synthetic_bias_dat.read_bytes()
        assert not stale.exists()

        # Archive without bias.dat
        bad = tmp_path / "bad.tgz"
        with tarfile.open(bad, "w:gz") as tar:
            tar.add(readme, arcname="README.txt")
        with pytest.raises(ValueError, match="No bias.dat member"):
            install_efcc18_bias_table(bad, verify_checksum=False)
        with pytest.raises(FileNotFoundError):
            install_efcc18_bias_table(tmp_path / "missing.tgz")

    def test_download_uses_requests_and_installs(
        self,
        isolated_env: Path,
        synthetic_bias_dat: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            tar.add(synthetic_bias_dat, arcname="bias.dat")
        payload = buffer.getvalue()
        seen: dict[str, Any] = {}

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *exc: Any) -> None:
                return None

            def raise_for_status(self) -> None:
                return None

            def iter_content(self, chunk_size: int) -> Any:
                for i in range(0, len(payload), chunk_size):
                    yield payload[i : i + chunk_size]

        def fake_get(url: str, stream: bool, timeout: float) -> FakeResponse:
            seen.update(url=url, stream=stream, timeout=timeout)
            return FakeResponse()

        import requests

        monkeypatch.setattr(requests, "get", fake_get)
        installed = download_efcc18_bias_table(verify_checksum=False, timeout_s=12.5)
        assert seen == {"url": EFCC18_ARCHIVE_URL, "stream": True, "timeout": 12.5}
        assert installed == isolated_env / "bias.dat"
        assert installed.read_bytes() == synthetic_bias_dat.read_bytes()


def _real_bias_dat_available() -> bool:
    try:
        resolve_bias_dat()
    except FileNotFoundError:
        return False
    return True


@pytest.mark.skipif(
    not _real_bias_dat_available(),
    reason="EFCC18 bias.dat not installed (pip install jpl-debias-2018)",
)
def test_real_bias_dat_integrity_and_parse(tmp_path: Path) -> None:
    """With the published table present: checksum, version tag and a parsed spot check."""
    from ..efcc18 import _sha256

    path = resolve_bias_dat()
    assert _sha256(path) == EFCC18_BIAS_DAT_SHA256
    assert read_efcc18_bias_version(path) == EFCC18_BIAS_VERSION
    table = load_efcc18_biases(path, cache_npy=tmp_path / "bias.npy")
    assert table.shape == (EFCC18_N_TILES, EFCC18_N_CATALOGS, 4)
    assert np.all(np.isfinite(table))
    # Corrections are sub-arcsec to a few arcsec; proper motions < 1 arcsec/yr.
    assert np.max(np.abs(table[:, :, :2])) < 10.0
    assert np.max(np.abs(table[:, :, 2:])) < 1000.0
    # Spot check: the k-th data line of the text file is tile k.
    k = 12345
    with path.open() as f:
        data_lines = (line for line in f if line.strip() and not line.startswith("!"))
        for _ in range(k):
            next(data_lines)
        row = np.array(next(data_lines).split(), dtype=np.float32)
    np.testing.assert_array_equal(table[k].reshape(-1), row)
