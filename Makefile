CC=g++
CFLAGS=--std=c++11

all:
	$(CC) $(CFLAGS) -o cpusdmm main.cpp


clean:
	rm -f cpusdmm
