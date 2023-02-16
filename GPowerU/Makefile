#Compile for debug or production 
DEBUG=n

#GPU Architecture
KEPLER=n
PASCAL=n
VOLTA=n
TURING=y
AMPERE=n

CUDA_INSTALL_PATH ?= /usr/local/cuda

CPPFLAGS = -D_REENTRANT=1 -D_XOPEN_SOURCE=600 \
	   -I $(CUDA_INSTALL_PATH)/include \
           -I ./include
CC = gcc
CXX = g++
NVCC	:= $(CUDA_INSTALL_PATH)/bin/nvcc
NVCCFLAGS :=  -ccbin $(CXX) -m64 --ptxas-options=-v  -maxrregcount=255 
#ROOTFLAGS := -I/usr/include/root -L/usr/lib64/root -lCore -lImt -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lROOTVecOps -lTree -lTreePlayer -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lMultiProc -lROOTDataFrame -lm -ldl


LIBS = -lcuda -lcublas -lcudart -lnvidia-ml -lpthread -std=c++11 

ifeq ($(DEBUG), y)
	CPPFLAGS += -DDEBUG
	CFLAGS   += -g
	NVCCFLAGS += -G -g -lineinfo
else
	CFLAGS   += -O2
endif


ifeq ($(KEPLER), y)
  SMVERSIONFLAGS +=  -gencode arch=compute_35,code=sm_35
endif

ifeq ($(PASCAL), y)
    SMVERSIONFLAGS +=  -gencode arch=compute_60,code=sm_60 
endif

ifeq ($(VOLTA), y)
    SMVERSIONFLAGS +=  -gencode arch=compute_70,code=sm_70 
endif

ifeq ($(TURING), y)
    SMVERSIONFLAGS +=  -gencode arch=compute_75,code=sm_75 
endif

ifeq ($(AMPERE), y)
    SMVERSIONFLAGS +=  -gencode arch=compute_75,code=sm_75 
endif


CUSRCS = powmeas.cu
APPS= powmeas

BINS= $(CSRCS:%.c=%) $(CUSRCS:%.cu=%)

all: .depend.cu $(APPS)


powmeas: powmeas.cu
	$(NVCC) $(NVCCFLAGS) $(ROOTFLAGS) $(CPPFLAGS)  $(SMVERSIONFLAGS) -o $@  $< $(LIBS)


.PHONY: clean
clean:
	rm -f .depend* *.o $(APPS) $(BINS)
	#rm *.csv
	#rm *.pdf
	rm data -r 

.PHONY: depend, dep,

#depend dep: .depend .depend.cu
depend dep: .depend.cu

#.depend:
#	$(CC) $(CFLAGS) $(CPPFLAGS) -MM $(CSRCS) > .depend

.depend.cu: $(CUSRCS)
	@if [ ! -z "$(CUSRCS)" ]; then ( $(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -M $(CUSRCS) > .depend.cu ); fi

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(ROOTFLAGS) $(SMVERSIONFLAGS) -o $@ -c $<


ifeq (.depend, $(wildcard .depend))
include .depend
endif
