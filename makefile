CC = /opt/rh/gcc-toolset-12/root/usr/bin/mpic++

CFLAGS = --std=c++11  -fmodules-ts  -fpermissive
LIB = -L/opt/rh/gcc-toolset-12/root/usr/lib -L/opt/rh/gcc-toolset-12/root/usr/lib/openmpi/lib -lmpi -lfftw3
INC = -I/usr/local/include -I/usr/include

test : test.cpp
	$(CC) $(CFLAGS) $^ -o $@  $(INC)  $(LIB) 