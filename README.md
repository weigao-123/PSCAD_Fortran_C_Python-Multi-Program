# PSCAD_Fortran_C_Python-Multi-Program
A simple case for PSCAD-Python program via C and Fortran

# Why this?
A project of power system needs to be simulated in PSCAD, and it utilizes some advanced Refincement Learning algorithms to control the simulation. Since PSCAD does not provide a API that can some advanced languages like Python can directly use.

# Basic structure
PSCAD can create a self-defined component, in which a like-Fortran code can be written to call a Fortran-C interface function, and then the C funtion can call Python to use the RL algorithm written in Python. The key is how to transfer the data between those softwares.

# Document introduction
A typic second order system, refer to PSCAD_DDPG.pscx
The Fortran-C-interface function, refer to fortran_interface_c_1.f90
The C-Python-interface function, refer to c_interface_python.c
The main RL algorithm, refer to ddpg_ain.py

# Environment setting
The multiple programing is based on PSCAD 4.6.3, Intel Fortran compiler, VS 2019, Python 3.7

Importrant thing 1: c_interface_python.c code needs some python include files and lib files (this include and lib folders are from Python root directory), since PSCAD will call VS 2019 via command line, we need to set the environment variables first, refer to: 
## https://github.com/MicrosoftDocs/cpp-docs.zh-cn/blob/live/docs/build/setting-the-path-and-environment-variables-for-command-line-builds.md
## https://blog.csdn.net/m0_38125278/article/details/87191971?utm_medium=distribute.pc_relevant_bbs_down.none-task--2~all~first_rank_v2~rank_v28-3.nonecase&depth_1-utm_source=distribute.pc_relevant_bbs_down.none-task--2~all~first_rank_v2~rank_v28-3.nonecase

Importrant thing 2: When C is calling Python, it needs to use a embeded Python interpreter which requires some Pyhton dependencies: Lib, DLLs, python37.dll (those folders are from Python root directory)

# Progress:
Tested C can call Python can pass data back and forth without problem, but with PSCAD, it seems like there are something incompatible.

(Put this aside)
