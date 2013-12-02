#! /bin/bash

gcc -I/usr/include/python2.7 geoutilsmodule.c -lpython2.7 -shared -fPIC -O3 -o ../openquake/hazardlib/geo/_utils_speedups.so
gcc -I/usr/include/python2.7 geodeticmodule.c -lpython2.7 -shared -fPIC -O3 -o ../openquake/hazardlib/geo/_geodetic_speedups.so
