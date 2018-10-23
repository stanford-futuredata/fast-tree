CC=clang++
CFLAGS=-std=c++14 -march=native -Wall -Wno-unused-function
INCLUDES=
TARGET=fast-tree

.PHONY: all debug asm clean

all:
	$(CC) $(CFLAGS) -O3 $(INCLUDES) main.cpp -o $(TARGET)


debug:
	$(CC) $(CFLAGS) -O0 -g -D_GLIBCXX_DEBUG -DDEBUG $(INCLUDES) main.cpp -o $(TARGET)

asm:
	$(CC) $(CFLAGS) $(INCLUDES) main.cpp -S

clean:
	rm -rf $(TARGET) main.dSYM *.s
