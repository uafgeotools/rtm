import os
import sys
sys.path.insert(0, os.path.abspath('../rtm'))

project = 'rtm'

copyright = '2020, David Fee & Liam Toney'

author = 'David Fee & Liam Toney'

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon',
              'sphinx.ext.intersphinx', 'recommonmark', 'sphinx.ext.viewcode']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'

napoleon_numpy_docstring = False

master_doc = 'index'
