CC=clang++
CFLAGS=-std=c++11 -march=native -Wall -Wno-unused-function
INCLUDES=
TARGET=main

.PHONY: all debug asm clean

all:
	$(CC) $(CFLAGS) -O3 $(INCLUDES) main.cpp -o $(TARGET)


debug:
	$(CC) $(CFLAGS) -O0 -g -DDEBUG $(INCLUDES) main.cpp -o $(TARGET)

asm:
	$(CC) $(CFLAGS) $(INCLUDES) main.cpp -S

clean:
	rm -rf $(TARGET) main.dSYM *.s
