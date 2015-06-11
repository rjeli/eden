CC=gcc-4.9 
CFLAGS=-fopenmp -std=c99 -pedantic -Wall -g -O3

all: 
	$(CC) $(CFLAGS) main.c -o main 

re: all
	./main
