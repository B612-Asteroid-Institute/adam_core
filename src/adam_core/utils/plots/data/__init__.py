from importlib import resources
from pathlib import Path

# Get the package resources for the logos directory
_data_path = resources.files("adam_core.utils.plots.data")

# Define paths to logo files
Coastlines = Path(_data_path / "ne_110m_coastline.zip")

__all__ = ["Coastlines"]
