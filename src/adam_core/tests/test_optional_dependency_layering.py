import subprocess
import sys
import textwrap


def test_normal_runtime_imports_do_not_load_astropy_or_astroquery() -> None:
    script = textwrap.dedent("""
        import sys

        import adam_core
        import adam_core.missions.porkchop
        import adam_core.observers
        import adam_core.orbits.query.sbdb
        import adam_core.time
        import adam_core.utils.mpc

        loaded = sorted(
            name
            for name in sys.modules
            if name == "astropy"
            or name.startswith("astropy.")
            or name == "astroquery"
            or name.startswith("astroquery.")
        )
        if loaded:
            raise AssertionError(f"optional providers loaded during normal imports: {loaded}")
        """)
    subprocess.run([sys.executable, "-c", script], check=True)
