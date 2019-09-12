rtm
===

rtm is a Python package for reverse time migration (RTM) of infrasound signals. Infrasound waveform data are back-projected over a grid of trial source locations. Based upon previous work by Sanderson et al. (in review) and Walker et al. (2010), this implementation is flexible and applicable to a wide variety of network geometries and sizes. Realistic travel times can be incorporated from numerical modeling or atmospheric specifications.

**References**

Sanderson, R., Matoza, R. S., Fee, D., Haney, M. M., & Lyons, J. J. (in review). Remote detection and location of explosive volcanism in Alaska with the EarthScope Transportable Array. _Journal of Geophysical Research: Solid Earth_.

Walker, K. T., Hedlin, M. A. H., de Groot‐Hedlin, C., Vergoz, J., Le Pichon, A., & Drob, D. P. (2010). Source location of the 19 February 2008 Oregon bolide using seismic networks and infrasound arrays. _Journal of Geophysical Research: Solid Earth_, 115, B12329. https://doi.org/10.1029/2010JB007863

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
conda create -n rtm -c conda-forge -c conda-forge/label/dev cartopy fiona gmt=6 obspy utm xarray
```

Usage
-----

To use rtm, clone or download this repository and add it to your Python path,
e.g. in a script where you'd like to use rtm:
```python
import sys
sys.path.insert(0, '/path/to/rtm')
```
Then you can access the package's functions with (for example)
```python
from rtm import define_grid
```
and so on. Currently, documentation only exists in function docstrings. For a
usage example, see [`example.py`](example.py).

Authors
-------

(_Alphabetical order by last name._)

David Fee  
Liam Toney
