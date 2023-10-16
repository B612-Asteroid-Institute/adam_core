# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'adam_core'
copyright = '2023, B612 Asteroid Institute'
author = 'B612 Asteroid Institute'

import adam_core
version = adam_core.__version__
release = adam_core.__version__

import adam_core.constants

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "quivr_sphinx_autodoc",
]

# From sphinx_toolbox.more_autodoc.typehints
hide_none_rtype = True

templates_path = ['_templates']
exclude_patterns = []

autodoc_type_aliases = {}

autodoc_member_order = 'bysource'
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pyarrow": ("https://arrow.apache.org/docs/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'furo'
# html_static_path = ['_static']

# nitpicky = True
# nitpick_ignore = {
#     ("py:mod", "quivr.columns"),
#     ("py:mod", "quivr.tables"),
#     # see: https://github.com/apache/arrow/issues/35413, should be fixed in 13.0.0
#     ("py:class", "pyarrow.FloatArray"),
#     ("py:class", "pyarrow.HalfFloatArray"),
#     ("py:class", "pyarrow.DoubleArray"),

#     ("py:mod", "pyarrow.compute"),
# }

