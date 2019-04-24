#WBL 21 March 2009 $Revision: 1.2 $
#based on cuda/sdk/projects/quasirandomGenerator/Makefile

################################################################################
#
# Build script for project
#
################################################################################

CUDA_INSTALL_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0"

# Add source files here
EXECUTABLE	:= park-miller

# CUDA source files (compiled with cudacc)
CUFILES_sm_20   := park-miller.cu

# CUDA dependency files
CU_DEPS		:= \
	park-miller_kernel.cuh \
	park-miller_common.h \
	realtype.h

# C dependency files
C_DEPS	:= \
	park-miller_common.h


# C/C++ source files (compiled with gcc / c++)
CCFILES		:= park-miller.cpp park-miller_gold.cpp


################################################################################
# Rules and targets

include ../../common/common.mk
