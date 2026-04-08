CFLAGS = -fsanitize=address -pedantic -Wall -g

run: main
	./main

main: main.c matrix.h neuralnetwork.h types.h
	gcc $(CFLAGS) -o main main.c -lm
	
test: main.c matrix.h neuralnetwork.h types.h
	gcc $(CFLAGS) -O3 -ffast-math -o test main.c -lm
	./test
	
