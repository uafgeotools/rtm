import os
import sys
sys.path.insert(0, os.path.abspath('../rtm'))

project = 'rtm'

html_show_copyright = False

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.intersphinx',
              'recommonmark',
              'sphinx.ext.viewcode',
              'sphinxcontrib.apidoc',
              'sphinx.ext.mathjax'
              ]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'

napoleon_numpy_docstring = False

master_doc = 'index'

autodoc_mock_imports = ['numpy',
                        'matplotlib',
                        'xarray',
                        'cartopy',
                        'obspy',
                        'scipy',
                        'mpl_toolkits'
                        ]

apidoc_module_dir = '../rtm'

apidoc_output_dir = 'api'

apidoc_separate_modules = True

apidoc_toc_file = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'obspy': ('https://docs.obspy.org/', None),
    'xarray': ('http://xarray.pydata.org/en/stable/', None),
    'cartopy': ('https://scitools.org.uk/cartopy/docs/latest/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'waveform_collection': ('https://uaf-waveform-collection.readthedocs.io/en/master/', None)
}
