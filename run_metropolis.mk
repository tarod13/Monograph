# run_metropolis.mk

all : spins.bin

spins.bin : header.h
	./header.h

header.h : metropolis.c
	gcc metropolis.c -lm -o header.h