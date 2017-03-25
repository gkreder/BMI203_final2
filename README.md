# Gabe Reder Final

[![Build
Status](https://travis-ci.org/gkreder/BMI203_final.svg?branch=master)](https://travis-ci.org/gkreder/BMI203_final)

## usage

To use the package, first run

```
conda install --yes --file requirements.txt
```

to install all the dependencies in `requirements.txt`. Then the package's
main function (located in `gabe_final/__main__.py`) can be run as
follows

```
python3 -m gabe_final
```

Functions are organized into .py files. __main__.py will run functions of interest, neural_net.py contains
the relevant functions for creating and running the neural_net. io.py contains relevant functions for
reading and parsing input data. 

## testing

Testing is as simple as running

```
python3 -m pytest
```

from the root directory of this project.


## contributors

Original design by Scott Pegg. Refactored and updated by Tamas Nagy.
