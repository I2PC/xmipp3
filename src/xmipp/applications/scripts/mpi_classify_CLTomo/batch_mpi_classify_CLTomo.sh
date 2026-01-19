#!/bin/sh
PROCS=$1
shift       
#PYTHONPATH=$XMIPP_HOME/binding/python:$PYTHONPATH
mpirun -np $PROCS $XMIPP_HOME/bin/xmipp_mpi_classify_CLTomo_prog "$@"
