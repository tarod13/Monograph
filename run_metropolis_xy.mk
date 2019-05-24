# run_metropolis_xy.mk

all : spins.bin

spins.bin : metropolis_xy.h
	./metropolis_xy.h

metropolis_xy.h : metropolis_xy.c
	gcc metropolis_xy.c -lm -o metropolis_xy.h