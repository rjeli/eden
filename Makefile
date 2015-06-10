CC=gcc-4.9 
CFLAGS=-fopenmp -std=c99 -pedantic -Wall

all: 
	$(CC) $(CFLAGS) main.c -o main 

re: all
	./main
