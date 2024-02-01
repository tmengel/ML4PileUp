# TennGen configuration file.
# Generated on Wed Dec 21 14:22:59 EST 2022 with the user supplied options:
# --cxx=g++
# --with-root-config=/nfs/home/tmengel/.conda/envs/myhepenv/bin/root-config
# --with-pythia8-config=/nfs/home/tmengel/.conda/envs/myhepenv/bin/pythia8-config
# --with-pythia8-lib=/nfs/home/tmengel/.conda/envs/myhepenv/lib
# --with-pythia8-include=/nfs/home/tmengel/.conda/envs/myhepenv/include
# --with-fastjet3-config=/lustre/isaac/scratch/tmengel/jet-background-subtraction/src/install/fastjet/bin/fastjet-config
# --with-fastjet3-lib=/lustre/isaac/scratch/tmengel/jet-background-subtraction/src/install/fastjet/lib
# --with-fastjet3-include=/lustre/isaac/scratch/tmengel/jet-background-subtraction/src/install/fastjet/include
# --prefix=/lustre/isaac/scratch/tmengel/jet-background-subtraction/src/install/tenngen

# Install directory prefixes.
PREFIX_BIN=/lustre/isaac/scratch/tmengel/jet-background-subtraction/src/install/tenngen/bin
PREFIX_INCLUDE=/lustre/isaac/scratch/tmengel/jet-background-subtraction/src/install/tenngen/include
PREFIX_LIB=/lustre/isaac/scratch/tmengel/jet-background-subtraction/src/install/tenngen/lib
PREFIX_SHARE=/lustre/isaac/scratch/tmengel/jet-background-subtraction/src/install/tenngen/share/TennGen200

# Compilation flags (see ./configure --help for further documentation).
CXX=g++
CXX_COMMON=-O2 -std=c++11 -pedantic -W -Wall -Wshadow -fPIC -pthread
CXX_SHARED=-shared
CXX_SONAME=-Wl,-soname,
LIB_SUFFIX=.so
OBJ_COMMON=

FASTJET3_USE=true
FASTJET3_CONFIG=/lustre/isaac/scratch/tmengel/jet-background-subtraction/src/install/fastjet/bin
FASTJET3_BIN=
FASTJET3_INCLUDE=-I/lustre/isaac/scratch/tmengel/jet-background-subtraction/src/install/fastjet/include
FASTJET3_LIB=-L/lustre/isaac/scratch/tmengel/jet-background-subtraction/src/install/fastjet/lib -Wl,-rpath,/lustre/isaac/scratch/tmengel/jet-background-subtraction/src/install/fastjet/lib -lfastjet

ROOT_USE=true
ROOT_CONFIG=/nfs/home/tmengel/.conda/envs/myhepenv/bin/root-config
ROOT_BIN=/nfs/home/tmengel/.conda/envs/myhepenv/bin/
ROOT_INCLUDE=-I/lustre/isaac/scratch/tmengel/.conda/envs/myhepenv/include
ROOT_LIB=-L/lustre/isaac/scratch/tmengel/.conda/envs/myhepenv/lib -Wl,-rpath,/lustre/isaac/scratch/tmengel/.conda/envs/myhepenv/lib -lCore

PYTHIA8_USE=true
PYTHIA8_CONFIG=/nfs/home/tmengel/.conda/envs/myhepenv/bin/pythia8-config
PYTHIA8_BIN=/lustre/isaac/scratch/tmengel/.conda/envs/myhepenv/bin/
PYTHIA8_INCLUDE=-I/lustre/isaac/scratch/tmengel/.conda/envs/myhepenv/include
PYTHIA8_LIB=-L/lustre/isaac/scratch/tmengel/.conda/envs/myhepenv/lib -Wl,-rpath,/lustre/isaac/scratch/tmengel/.conda/envs/myhepenv/lib
