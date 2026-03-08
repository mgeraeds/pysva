# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Make project root importable for autodoc
sys.path.insert(0, os.path.abspath('../..'))

project = 'pySVA'
copyright = '2026, M. Geraeds'
author = 'M. Geraeds'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.napoleon',
	'sphinx.ext.viewcode',
]

# Mock imports that may not be available in the doc build environment
autodoc_mock_imports = [
	'dfmproc', 'dfm_tools', 'xugrid', 'xarray', 'numpy', 'dask', 'dfm_tools.xugrid_helpers', 'dfmproc'
]

# Autodoc options
autodoc_default_options = {
	'members': True,
	'undoc-members': True,
	'show-inheritance': True,
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_sidebars = {
	'**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html']
}
