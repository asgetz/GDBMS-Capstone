#WBL 4 April 2009 $Revision: 1.3 $

#WBL 31 Mar 2011 Add own common.mk rather than using one in use

mv cuda/sdk/common/common.mk cuda/sdk/common/common.mk_save
cp -p cuda/sdk/projects/park-miller/park-miller_common.mk \
      cuda/sdk/common/common.mk

tar cvf ~/cuda_park-miller.tar				\
  cuda/sdk/projects/park-miller/README.txt		\
  cuda/sdk/projects/park-miller/make_kit.bat		\
\
  cuda/sdk/projects/park-miller/Makefile		\
  cuda/sdk/common/common.mk				\
\
  cuda/sdk/projects/park-miller/park-miller.cpp		\
  cuda/sdk/projects/park-miller/park-miller.cu		\
  cuda/sdk/projects/park-miller/park-miller_gold.cpp	\
  cuda/sdk/projects/park-miller/park-miller_kernel.cuh	\
\
  cuda/sdk/projects/park-miller/realtype.h		\
  cuda/sdk/projects/park-miller/park-miller_common.h	\
\
  cuda/sdk/projects/park-miller/example.bat		\
  cuda/sdk/projects/park-miller/example.log		\

mv cuda/sdk/common/common.mk_save cuda/sdk/common/common.mk

gzip ~/cuda_park-miller.tar

#

