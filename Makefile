
run: main
	./main

main: main.c matrix.h neuralnetwork.h types.h
	gcc -fsanitize=address -pedantic -Wall -Wextra -g -o main main.c -lm
	
test: main.c matrix.h neuralnetwork.h types.h
	gcc -O3 -fsanitize=address -pedantic -Wall -Wextra -g -o test main.c -lm
	./test
	
