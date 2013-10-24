#!/bin/bash

echo "inside $0"

## CONSTANTS

PIP_ARGS="-IUM"
# -I ignore installed
# -U upgrade package
# -M use PyPI mirrors

#PIP_ARGS+="q"
# -q quiet

## HELPERS

# install a package via pip without heavy hitting he line space economy
function pip_install()
{
    echo "----------------------------------------"
    time pip install $PIP_ARGS $1 2>&1 | tail -fn2
}

## UBUNTU INSTALL

case ${TRAVIS_PYTHON_VERSION:0:3} in
    2.6)
        time sudo apt-get install -qq python-dev libatlas-dev libatlas-base-dev liblapack-dev gfortran 2>&1 | trial -n2;
        time sudo apt-get install -qq python-dev python-numpy python-numpy-dev python-scipy 2>&1 | tail -n2 ;;
    2.7)
        time sudo apt-get install -qq python-dev python-numpy python-numpy-dev python-scipy 2>&1 | tail -n2 ;;
    3.2)
        exit false ;;
    3.3)
        exit false ;;
esac

# force virtualenv to accept system_site_packages
#rm -f $VIRTUAL_ENV/lib/python$TRAVIS_PYTHON_VERSION/no-global-site-packages.txt

## PIP INSTALL

# need setuptools?
#pip_install setuptools

# install requirements
pip_install numpy
pip_install scipy
pip_install scikit-learn
pip_install MDP
pip_install pyyaml

# build and install
time python setup.py build_ext install

## EOF
true
