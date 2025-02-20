import base64
from importlib import resources
from pathlib import Path

# Get the package resources for the logos directory
_logos_path = resources.files("adam_core.utils.plots.logos")

# Define paths to logo files
AsteroidInstituteLogoDark = Path(
    _logos_path / "AsteroidInstituteProgramForDark_Large.png"
)
AsteroidInstituteLogoLight = Path(
    _logos_path / "AsteroidInstituteProgramTransparent_Large.png"
)


def get_logo_base64(logo_path: Path) -> str:
    """Convert logo image to base64 string.

    Parameters
    ----------
    logo_path : Path
        Path to the logo image file

    Returns
    -------
    str
        Base64 encoded image data with data URL prefix
    """
    with open(logo_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


__all__ = ["AsteroidInstituteLogoDark", "AsteroidInstituteLogoLight", "get_logo_base64"]
