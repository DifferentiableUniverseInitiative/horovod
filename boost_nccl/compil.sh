#!/bin/bash

echo Compiling $1


g++ -Wall -shared -fPIC -o $1.so $1.cpp -std=c++0x -lboost_python38 -I/gpfslocalsup/pub/anaconda-py3/2020.11/include/python3.8/ -L/gpfswork/idris/hpe/shpe033/.conda/envs/mtf/lib/python3.8/site-packages/horovod-0.21.1-py3.8-linux-x86_64.egg/horovod/tensorflow

#conda activate mtf
#
#module load boost/1.70.0
#
#export C_INCLUDE_PATH=$C_INCLUDE_PATH:/gpfslocalsup/spack_soft/python/3.8.6/gcc-8.3.1-rmwrkoturjccinljag6fwndw7fsgpepg/include/python3.8:
#export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/gpfslocalsup/spack_soft/python/3.8.6/gcc-8.3.1-rmwrkoturjccinljag6fwndw7fsgpepg/include/python3.8
#
#export LIBRARY_PATH=/gpfslocalsup/spack_soft/boost/1.70.0/gcc-8.3.1-ac7s6bl2rvxgv7kt5dhqujijptqob26c/lib:/gpfswork/idris/hpe/shpe033/meshtensorflow_project/test/boost:/gpfslocalsup/spack_soft/gcc/7.3.0/gcc-8.3.1-vqzoua4fyg6e5jiz3vhkpjb4qtofjfrf/lib64:/gpfslocalsup/spack_soft/gcc/7.3.0/gcc-8.3.1-vqzoua4fyg6e5jiz3vhkpjb4qtofjfrf/lib:/gpfslocalsup/spack_soft/boost/1.70.0/intel-19.0.4-5zoh2xvpvjl3ecgofb5t75zy2tgasuxd/lib:/gpfslocalsys/intel/parallel_studio_xe_2019_update4_cluster_edition/compilers_and_libraries_2019.4.243/linux/compiler/lib/intel64_lin:/gpfslocalsys/slurm/current/lib/slurm:/gpfslocalsys/slurm/current/lib
#
#gcc -Wall -shared -fPIC -lboost_python38 -lboost_numpy38 -o boost_nccl.so boost_nccl.cc
