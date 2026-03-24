
run: main
	./main

main: main.c
	gcc -pedantic -Wall -Wextra -g -o main main.c -lm