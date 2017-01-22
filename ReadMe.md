AutoXGBoost.py [![Unlicensed work](https://raw.githubusercontent.com/unlicense/unlicense.org/master/static/favicon.png)](https://unlicense.org/)
===============
[![PyPi Status](https://img.shields.io/pypi/v/AutoXGBoost.svg)](https://pypi.python.org/pypi/AutoXGBoost)
![GitLab Build Status](https://gitlab.com/KOLANICH/AutoXGBoost.py/badges/master/pipeline.svg)
[![TravisCI Build Status](https://travis-ci.org/KOLANICH/AutoXGBoost.py.svg?branch=master)](https://travis-ci.org/KOLANICH/AutoXGBoost.py)
![GitLab Coverage](https://gitlab.com/KOLANICH/AutoXGBoost.py/badges/master/coverage.svg)
[![Coveralls Coverage](https://img.shields.io/coveralls/KOLANICH/AutoXGBoost.py.svg )](https://coveralls.io/r/KOLANICH/AutoXGBoost.py)
[![Libraries.io Status](https://img.shields.io/librariesio/github/KOLANICH/AutoXGBoost.py.svg)](https://libraries.io/github/KOLANICH/AutoXGBoost.py)
[![Gitter.im](https://badges.gitter.im/AutoXGBoost.py/Lobby.svg)](https://gitter.im/AutoXGBoost.py/Lobby)

A **VERY** early alpha, don't use. Ore use, but refactor first and send a PR. This is here basically for myself. For now it was designed with imputation in mind, but I'm going to split imputation from it.

Features
========
* heavily automated and easy to use;
* serializes and deserializes XGBoost models;
* fits XGBoost models;
* optimizes hyperparams;
* predicts one column based on other columns.


Requirements
------------
* [```Python >=3.4```](https://www.python.org/downloads/). [```Python 2``` is dead, stop raping its corpse.](https://python3statement.org/) Use ```2to3``` with manual postprocessing to migrate incompatible code to ```3```. It shouldn't take so much time. For unit-testing you need Python 3.6+ or PyPy3 because their ```dict``` is ordered and deterministic.

* [```numpy```](https://github.com/numpy/numpy) ![Licence](https://img.shields.io/github/license/numpy/numpy.svg) [![PyPi Status](https://img.shields.io/pypi/v/numpy.svg)](https://pypi.python.org/pypi/numpy) [![TravisCI Build Status](https://travis-ci.org/numpy/numpy.svg?branch=master)](https://travis-ci.org/numpy/numpy) [![Libraries.io Status](https://img.shields.io/librariesio/github/numpy/numpy.svg)](https://libraries.io/github/numpy/numpy)

* [```scipy```](https://github.com/scipy/scipy) ![Licence](https://img.shields.io/github/license/scipy/scipy.svg) [![PyPi Status](https://img.shields.io/pypi/v/scipy.svg)](https://pypi.python.org/pypi/scipy) [![TravisCI Build Status](https://travis-ci.org/scipy/scipy.svg?branch=master)](https://travis-ci.org/scipy/scipy) [![CodeCov Coverage](https://codecov.io/github/scipy/scipy/coverage.svg?branch=master)](https://codecov.io/github/scipy/scipy/) [![Libraries.io Status](https://img.shields.io/librariesio/github/scipy/scipy.svg)](https://libraries.io/github/scipy/scipy)

* [```pandas```](https://github.com/pandas-dev/pandas) ![Licence](https://img.shields.io/github/license/pandas-dev/pandas.svg) [![PyPi Status](https://img.shields.io/pypi/v/pandas.svg)](https://pypi.python.org/pypi/pandas) [![TravisCI Build Status](https://travis-ci.org/pandas-dev/pandas.svg?branch=master)](https://travis-ci.org/pandas-dev/pandas) [![CodeCov Coverage](https://codecov.io/github/pandas-dev/pandas/coverage.svg?branch=master)](https://codecov.io/github/pandas-dev/pandas/) [![Libraries.io Status](https://img.shields.io/librariesio/github/pandas-dev/pandas.svg)](https://libraries.io/github/pandas-dev/pandas) [![Gitter.im](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/pydata/pandas)


* [```xgboost```](https://github.com/dmlc/xgboost) ![Licence](https://img.shields.io/github/license/dmlc/xgboost.svg) [![PyPi Status](https://img.shields.io/pypi/v/xgboost.svg)](https://pypi.python.org/pypi/xgboost) [![TravisCI Build Status](https://travis-ci.org/dmlc/xgboost.svg?branch=master)](https://travis-ci.org/dmlc/xgboost) [![Libraries.io Status](https://img.shields.io/librariesio/github/dmlc/xgboost.svg)](https://libraries.io/github/dmlc/xgboost) [![Gitter.im](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dmlc/xgboost)

* [```tqdm```](https://github.com/tqdm/tqdm) ![Licence](https://img.shields.io/github/license/tqdm/tqdm.svg) [![PyPi Status](https://img.shields.io/pypi/v/tqdm.svg)](https://pypi.python.org/pypi/tqdm) [![TravisCI Build Status](https://travis-ci.org/tqdm/tqdm.svg?branch=master)](https://travis-ci.org/tqdm/tqdm) [![Coveralls Coverage](https://img.shields.io/coveralls/tqdm/tqdm.svg)](https://coveralls.io/r/tqdm/tqdm) [![CodeCov Coverage](https://codecov.io/github/tqdm/tqdm/coverage.svg?branch=master)](https://codecov.io/github/tqdm/tqdm/) [![Codacy Grade](https://api.codacy.com/project/badge/Grade/3f965571598f44549c7818f29cdcf177)](https://www.codacy.com/app/tqdm/tqdm) [![Libraries.io Status](https://img.shields.io/librariesio/github/tqdm/tqdm.svg)](https://libraries.io/github/tqdm/tqdm)

* [```Chassis.py```](https://github.com/KOLANICH/Chassis.py) ![Licence](https://img.shields.io/github/license/KOLANICH/Chassis.py.svg) [![PyPi Status](https://img.shields.io/pypi/v/Chassis.py.svg)](https://pypi.python.org/pypi/Chassis.py)
[![TravisCI Build Status](https://travis-ci.org/KOLANICH/Chassis.py.svg?branch=master)](https://travis-ci.org/KOLANICH/Chassis.py)
[![Coveralls Coverage](https://img.shields.io/coveralls/KOLANICH/Chassis.py.svg)](https://coveralls.io/r/KOLANICH/Chassis.py)
[![Libraries.io Status](https://img.shields.io/librariesio/github/KOLANICH/Chassis.py.svg)](https://libraries.io/github/KOLANICH/Chassis.py)
[![Gitter.im](https://badges.gitter.im/Chassis.py/Lobby.svg)](https://gitter.im/Chassis.py/Lobby)
