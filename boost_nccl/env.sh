module load python/3.8.2
export PYTHONPATH=/gpfslocalsup/pub/anaconda-py3/2020.11/bin/python
# boost compile avec python
export CPATH=/gpfslocalsup/spack_soft/boost/1.70.0/gcc-8.3.1-ac7s6bl2rvxgv7kt5dhqujijptqob26c/include:$CPATH
export LD_LIBRARY_PATH=/gpfslocalsup/spack_soft/boost/1.70.0/gcc-8.3.1-ac7s6bl2rvxgv7kt5dhqujijptqob26c/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/gpfslocalsup/spack_soft/boost/1.70.0/gcc-8.3.1-ac7s6bl2rvxgv7kt5dhqujijptqob26c/lib:$LIBRARY_PATH
export CMAKE_PREFIX_PATH=/gpfslocalsup/spack_soft/boost/1.70.0/gcc-8.3.1-ac7s6bl2rvxgv7kt5dhqujijptqob26c/
export BOOST_ROOT=/gpfslocalsup/spack_soft/boost/1.70.0/gcc-8.3.1-ac7s6bl2rvxgv7kt5dhqujij




#module load flatbuffers/1.11.0
#module load boost/1.70.0
#module load tensorflow-gpu/py3/2.3.1
#
#export LD_LIBRARY_PATH=/gpfswork/idris/hpe/shpe033/horovod-v0.21.1/horovod/tensorflow:/gpfslocalsup/spack_soft/boost/1.70.0/gcc-8.3.1-ac7s6bl2rvxgv7kt5dhqujijptqob26c/lib:/gpfswork/idris/hpe/shpe033/meshtensorflow_project/boost_nccl:/gpfswork/idris/hpe/shpe033/meshtensorflow_project/install_env/cuda:/gpfslocalsup/spack_soft/openmpi/4.0.2/intel-19.0.4-q6fk6qb3intsc3raxyvu6x3as6uadzsl/lib:/gpfslocalsup/spack_soft/cudnn/7.6.5.32-10.1-linux-x64/gcc-4.8.5-fn27wz3xidimpfcu4t3ctvc6vxjr3afy/lib64:/gpfslocalsup/spack_soft/nccl/2.5.6-2/gcc-4.8.5-qtdldqth7z3ybxfozhgrjryw6c2ideaw/lib:/gpfslocalsys/cuda/10.1.2/nvvm/lib64:/gpfslocalsys/cuda/10.1.2/extras/CUPTI/lib64:/gpfslocalsys/cuda/10.1.2/lib64:/gpfslocalsys/cuda/10.1.2/samples/common/lib/linux/x86_64:/gpfslocalsys/cuda/10.1.2/targets/x86_64-linux/lib:/gpfslocalsup/spack_soft/boost/1.70.0/intel-19.0.4-5zoh2xvpvjl3ecgofb5t75zy2tgasuxd/lib:/gpfslocalsys/intel/parallel_studio_xe_2019_update4_cluster_edition/compilers_and_libraries_2019.4.243/linux/compiler/lib/intel64_lin:/gpfslocalsup/spack_soft/flatbuffers/1.11.0/gcc-9.1.0-jtgrepiqbrbzlsjawqlmprdmcqp5drqu/lib64:/gpfslocalsup/spack_soft/gcc/9.1.0/gcc-8.3.1-dsq3humdshff2skbethmwa2pg4s2f7rz/lib64:/gpfslocalsup/spack_soft/gcc/9.1.0/gcc-8.3.1-dsq3humdshff2skbethmwa2pg4s2f7rz/lib:/gpfswork/idris/hpe/shpe033/horovod-v0.21.1/horovod/tensorflow:/gpfslocalsup/spack_soft/boost/1.70.0/gcc-8.3.1-ac7s6bl2rvxgv7kt5dhqujijptqob26c/lib:/gpfswork/idris/hpe/shpe033/meshtensorflow_project/boost_nccl:/gpfswork/idris/hpe/shpe033/meshtensorflow_project/install_env/cuda:/gpfswork/idris/hpe/shpe033/horovod-v0.21.1/horovod/tensorflow:/gpfslocalsup/spack_soft/boost/1.70.0/gcc-8.3.1-ac7s6bl2rvxgv7kt5dhqujijptqob26c/lib:/gpfswork/idris/hpe/shpe033/meshtensorflow_project/boost_nccl:/gpfswork/idris/hpe/shpe033/meshtensorflow_project/install_env/cuda:/gpfswork/idris/hpe/shpe033/horovod-v0.21.1/horovod/tensorflow:/gpfslocalsup/spack_soft/boost/1.70.0/gcc-8.3.1-ac7s6bl2rvxgv7kt5dhqujijptqob26c/lib:/gpfswork/idris/hpe/shpe033/meshtensorflow_project/boost_nccl:/gpfswork/idris/hpe/shpe033/meshtensorflow_project/install_env/cuda:/gpfswork/idris/hpe/shpe033/meshtensorflow_project/boost_nccl:/gpfswork/idris/hpe/shpe033/meshtensorflow_project/boost_nccl:/gpfswork/idris/hpe/shpe033/horovod-v0.21.1/horovod/tensorflow:/gpfslocalsup/spack_soft/boost/1.70.0/gcc-8.3.1-ac7s6bl2rvxgv7kt5dhqujijptqob26c/lib:/gpfswork/idris/hpe/shpe033/meshtensorflow_project/boost_nccl:/gpfswork/idris/hpe/shpe033/meshtensorflow_project/install_env/cuda:
#
