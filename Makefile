CC=g++
CFLAGS=--std=c++11
#CFLAGS=--std=c++11 -DFDEBUG

all:
	$(CC) $(CFLAGS) -o cpusdmm main.cpp


clean:
	rm -f cpusdmm
