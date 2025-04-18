#Compile for debug or production 
DEBUG=n

#GPU Architecture
KEPLER=n
PASCAL=n
VOLTA=n
TURING=n
AMPERE=y

CUDA_INSTALL_PATH ?= /usr/local/cuda

CPPFLAGS = -D_REENTRANT=1 -D_XOPEN_SOURCE=600 \
	   -I $(CUDA_INSTALL_PATH)/include \
           -I ./include
CC = gcc
CXX = g++
NVCC	:= nvcc  #$(CUDA_INSTALL_PATH)/bin/nvcc
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
    SMVERSIONFLAGS +=  -gencode arch=compute_80,code=sm_80
endif


CUSRCS = powmeas.cu
APPS= powmeas

DEPS := $(patsubst %,%.d,$(basename $(CUSRCS)))

BINS= $(CSRCS:%.c=%) $(CUSRCS:%.cu=%)

all: $(APPS)


powmeas.o: powmeas.cu powmeas.d 
	$(NVCC) $(NVCCFLAGS) $(ROOTFLAGS) $(CPPFLAGS)  $(SMVERSIONFLAGS) -c $@  $< $(LIBS)

powmeas: powmeas.o
	$(NVCC) $(NVCCFLAGS) $(ROOTFLAGS) $(CPPFLAGS)  $(SMVERSIONFLAGS) -o $@  $< $(LIBS)


#.PHONY: clean
clean:
	$(RM)  *.d *.o $(APPS) $(BINS)
	$(RM) data -r 

.PHONY: powmeas, clean, all
powmeas.d: $(CUSRCS)
	@if [ ! -z "$(CUSRCS)" ]; then ( $(NVCC) $(CPPFLAGS) -M $(CUSRCS) > $@ ); fi

%.o: %.cu %.d
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) $(SMVERSIONFLAGS) -o $@ -c $<

-include $(DEPS)
