import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import solax  # noqa: E402

project = "quantumsolax"
copyright = "Pavlo Bilous"
author = "Pavlo Bilous"
release = solax.__version__
# The default title is "<project> v<release> documentation" -- release is
# a long git-derived dev-build string (e.g. "0.1.dev36+g2e5a382c9") between
# tagged versions, which wraps awkwardly in the sidebar. Keep the title
# short and stable; the full version is still available via solax.__version__.
html_title = f"{project} documentation"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    # solax's __init__.py files re-export their public API via
    # `from .submodule import *` rather than defining names directly, and
    # declare no __all__ -- without this, autodoc's automodule directive
    # silently omits every re-exported class/function.
    "imported-members": True,
    # Dunder methods (__add__, __eq__, __call__, ...) carry most of the
    # actual documentation on solax's core classes; autodoc hides them by
    # default.
    "special-members": "__init__, __call__, __add__, __sub__, __mul__, "
    "__truediv__, __neg__, __eq__, __len__, __getitem__, __str__",
}

templates_path = []
exclude_patterns = ["_build"]

html_theme = "furo"
html_static_path = []
