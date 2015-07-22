#
CC=nvc
LD=cl
CFLAGS= -c -O3 -I$(CUDA_INC_PATH) -I$(VVEDIR)\include\vvelib -I$(QTINC) --host-compilation C++ --ptxas-options=-v
LDFLAGS= /LD $(CUDA_LIB_PATH)\cudart.lib

cuda.obj: cuda.cu vector.h
	$(CC) $(CFLAGS) -o cuda.obj cuda.cu

cuda.dll: cuda.obj
	$(LD) /LD cuda.obj C:\CUDA\LIB\cudart.lib

