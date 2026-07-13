import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path


def test_default_dependency_metadata_keeps_optional_providers_out() -> None:
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    dependencies = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    normalized = {dependency.lower().replace("_", "-") for dependency in dependencies}

    assert not any(dependency.startswith("astropy") for dependency in normalized)
    assert not any(dependency.startswith("astroquery") for dependency in normalized)
    assert not any(dependency.startswith("healpy") for dependency in normalized)
    assert not any(dependency.startswith("timezonefinder") for dependency in normalized)
    assert not any(dependency.startswith("h3") for dependency in normalized)


def test_normal_runtime_imports_do_not_load_astropy_or_astroquery() -> None:
    script = textwrap.dedent("""
        import sys

        import adam_core
        import adam_core.missions.porkchop
        import adam_core.observers
        import adam_core.orbits.query.sbdb
        import adam_core.time
        import adam_core.utils.mpc

        optional_roots = {"astropy", "astroquery", "healpy", "timezonefinder", "h3"}
        loaded = sorted(
            name for name in sys.modules if name.split(".", 1)[0] in optional_roots
        )
        if loaded:
            raise AssertionError(f"optional providers loaded during normal imports: {loaded}")
        """)
    subprocess.run([sys.executable, "-c", script], check=True)
