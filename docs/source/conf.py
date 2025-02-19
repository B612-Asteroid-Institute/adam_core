# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "adam_core"
copyright = "2023, B612 Asteroid Institute"
author = "B612 Asteroid Institute"

import adam_core

version = adam_core.__version__
release = adam_core.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "myst_nb",
]

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = False
add_module_names = False

# Autodoc settings
autodoc_typehints = "description"
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__, _abc_impl, TypeVar',
    'inherited-members': False,
    'show-inheritance': True,
}
autodoc_member_order = "bysource"
autodoc_type_aliases = {
    "Coordinates": "adam_core.coordinates.types.Coordinates",
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

templates_path = ["_templates"]
exclude_patterns = []

import quivr

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "quivr": (f"https://quivr.readthedocs.io/en/v{quivr.__version__}", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2980B9",
        "color-brand-content": "#2980B9",
        "color-admonition-background": "#E1F5FE",
    },
    "dark_css_variables": {
        "color-brand-primary": "#56B4E9",
        "color-brand-content": "#56B4E9",
        "color-admonition-background": "#1A1A1A",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}
html_static_path = ['_static']
html_title = f"adam_core {version}"

# Optional: Configure nbsphinx settings
nbsphinx_execute = 'auto'  # Execute notebooks on build
nbsphinx_allow_errors = True  # Continue building even if cells error
