import os
import sys

sys.path.insert(0, os.path.abspath(".."))
project = "building-digital-twin"
author = "Mohamed Al-Mandhari"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode", "sphinx.ext.mathjax"]
autodoc_member_order = "bysource"
autodoc_mock_imports = ["numba"]
html_theme = "alabaster"
exclude_patterns = ["_build"]
