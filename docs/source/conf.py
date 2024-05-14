# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

#   How to build the documentation:
# cd docs
# pip install myst_nb pydata-sphinx-theme
# make html
# make latexpdf

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys
sys.path.insert(0, os.path.abspath('../..'))

import rumdpy

project = 'rumdpy'
copyright = '2024, Thomas Schrøder, Ulf R. Pedersen, Rishabh Sharma, Lorenzo Costigliola'
author = 'Thomas Schrøder, Ulf R. Pedersen, Rishabh Sharma, Lorenzo Costigliola'
release = rumdpy.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',  # enable markdown files (*.md), and Jupyter Notebooks (*.ipynb)
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_logo = '_static/logo_777x147.png'