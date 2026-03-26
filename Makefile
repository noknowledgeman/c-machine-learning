
run: main
	./main

main: main.c matrix.h neuralnetwork.h types.h
	gcc -fsanitize=undefined -pedantic -Wall -Wextra -g -o main main.c -lm
