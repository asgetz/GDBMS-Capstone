WBL 4 April 2009 $Revision: 1.5 $
Write up: 
A Fast High Quality Pseudo Random Number Generator for nVidia CUDA, 
W. B. Langdon, CIGPU 2009, pp2511-2513, doi:10.1145/1570256.1570353

WBL 31 Mar 2011 Add section on errors
WBL  3 Aug 2009 Add comment from Eric Thibodeau on common.mk

Makefile and common.mk	compile example. Eg:
			Change ROOTDIR in common.mk
			setenv verbose 1 #show compilation commands
			make
			../../../bin/linux/release/park-miller

You may need to check for differences between common.mk and the one
installed on your system. On some system the additional line,
"NVCCFLAGS += -include=vararg-fix.h", is required to get the program
to compile.

park-miller.cpp		main

park-miller.cu		linking stub from PC main to CUDA Kernel

park-miller_gold.cpp	PC implementation of Park-Miller to check
			GPU's answers

park-miller_kernel.cuh	CUDA Kernel

park-miller_common.h	linking stub

realtype.h		nVidia

example.bat		Script with example command line
example.log		Output of example.bat

Common errors:
==============
mkdir: cannot create directory `/opt/cuda/sdk': Read-only file system
suggests problem with ROOTDIR or inconsistency with common.mk
eg try overriding ROOTDIR by adding something like the following to Makefile
ROOTDIR  := /my_directory/cuda/sdk

park-miller.cpp:44:26: error: cutil_inline.h: No such file or directory
Suggests problem with COMMONDIR or inconsistency with common.mk
eg try overriding ROOTDIR2 by adding something like the following to Makefile
ROOTDIR2 := /usr/local/cuda/NVIDIA_GPU_Computing_SDK/C/tools

/usr/bin/ld: cannot find -lcutil
Suggests problem with LIBDIR or inconsistency with common.mk
This can occur when there are multiple versions of CUDA installed.
eg try overriding LIBDIR by adding something like the following to Makefile
LIBDIR   := /my_directory/cuda_3.1/cuda/NVIDIA_CUDA_SDK/lib
However eventually it may be better to resolve the problem of multiple
version of CUDA and/or create your own make file or compilation
script or process. 

../../../bin/linux/release/park-miller: error while loading shared libraries: libcudart.so.2: cannot open shared object file: No such file or directory
LD_LIBRARY_PATH incorrectly defined.
Eg try 
setenv LD_LIBRARY_PATH "$LD_LIBRARY_PATH":/usr/opt/cuda/lib

park-miller.cpp:43:26: error: cuda_runtime.h: No such file or directory
Perhaps a problem with ROOTDIR2 however check really do have a copy
of cuda_runtime.h installed somewhere. 
Perhaps a problem with CUDA_INSTALL_PATH
eg try overriding CUDA_INSTALL_PATH by adding something like to Makefile
CUDA_INSTALL_PATH := /usr/local/cuda-3.0

/usr/bin/ld: skipping incompatible /usr/local/cuda-3.0/lib/libcudart.so when searching for -lcudart
Perhaps a 32 bit v 64 bit problem?
Perhaps try overriding LIBDIR by adding something like to Makefile
LIBDIR := /usr/local/cuda/lib64
