rtm
===

Reverse time migration of infrasound signals.

Dependencies
------------

* [cartopy](https://scitools.org.uk/cartopy/docs/latest/)
* [Fiona](https://fiona.readthedocs.io/en/latest/)
* [GMT](https://docs.generic-mapping-tools.org/dev/index.html)
* [ObsPy](http://docs.obspy.org/)
* [utm](https://github.com/Turbo87/utm)
* [xarray](http://xarray.pydata.org/en/stable/)

...and their dependencies, which you don't really have to be concerned about if
you're using [conda](https://docs.conda.io/projects/conda/en/latest/index.html)!

It's recommended that you create a new conda environment to use with this
repository:
```
conda create -n rtm -c conda-forge -c conda-forge/label/cf201901 cartopy fiona gmt=6 obspy utm xarray
```

Usage
-----

To use rtm, clone or download this repository and add it to your Python path,
e.g. in a script where you'd like to use rtm:
```python
import sys
sys.path.insert(0, '/path/to/rtm')
```
Then you can access the module's functions with (for example)
```python
from rtm import define_grid
```
and so on. Currently, documentation only exists in function docstrings. For a
usage example, see `example.py`.

Authors
-------

(_Alphabetical order by last name._)

David Fee  
Liam Toney
