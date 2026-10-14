# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import os
import sys

# Add src directory to path so we can import inverse
DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DOCS_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import inverse  # noqa: E402

# -- Project information -----------------------------------------------------

project = "Inverse"
copyright = ""
html_show_copyright = False
author = "Troy N. Smith"

# The full version, including alpha/beta/rc tags
release = inverse.__version__
version = release

# The suffix(es) of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

html_logo = "comms/FuelLibLogo_Blue.png"
# -- General configuration ---------------------------------------------------

numfig = True
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["refs.bib"]

# sphinx_autodoc_typehints

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "no-module": True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Autodoc type hints configuration
autoclass_content = "both"
autodoc_typehints = "description"
simplify_optional_unions = (
    False  # Show Optional[Union[A, B]] instead of Union[A, B, None]
)
typehints_use_signature = True  # Show parameter types in signature
typehints_use_signature_return = True  # Show return type in signature
typehints_defaults = "braces"  # "param (type, (default: value)) -- description"

# Explicitly tell Napoleon to parse Google docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False  # Turn off NumPy if you only use Google
napoleon_include_init_with_doc = True
napoleon_use_ivar = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets)
# html_static_path = ["_static"]
# html_css_files = ["custom.css"]

html_theme_options = {"collapse_navigation": False, "navigation_depth": 3}
