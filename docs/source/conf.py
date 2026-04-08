"""Sphinx configuration for adam_core documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from importlib import metadata

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

project = "adam_core"
author = "B612 Asteroid Institute"
copyright = f"2023-{datetime.now().year}, {author}"

try:
    import adam_core

    version = adam_core.__version__
    release = adam_core.__version__
except Exception:
    version = metadata.version("adam_core")
    release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_sitemap",
    "myst_nb",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "build",
    "Thumbs.db",
    ".DS_Store",
]

autosectionlabel_prefix_document = True

autodoc_member_order = "bysource"
autodoc_type_aliases = {
    "Coordinates": "adam_core.coordinates.types.Coordinates",
}

# Notebook policy: render notebooks in docs but do not execute in RTD/CI docs builds.
nb_execution_mode = "off"
nb_number_source_lines = True
nb_merge_streams = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "quivr": ("https://quivr.readthedocs.io/en/latest/", None),
}

html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "adam_core documentation"
html_baseurl = os.environ.get(
    "READTHEDOCS_CANONICAL_URL",
    "https://adam-core.readthedocs.io/en/latest/",
)
html_last_updated_fmt = "%Y-%m-%d"
nitpicky = False

html_theme_options = {
    "sidebar_hide_name": False,
    "top_of_page_button": "edit",
    "navigation_with_keys": True,
    "source_repository": "https://github.com/B612-Asteroid-Institute/adam_core/",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "light_css_variables": {
        "color-brand-primary": "#0f766e",
        "color-brand-content": "#0369a1",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5eead4",
        "color-brand-content": "#7dd3fc",
    },
}

sitemap_url_scheme = "{link}"
