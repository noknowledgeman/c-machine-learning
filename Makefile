CFLAGS = -fsanitize=address -pedantic -Wall -g
FILES = main.c matrix.h neuralnetwork.h types.h

run: main
	./main
	
run-test: test
	./test

main: $(FILES)
	gcc $(CFLAGS) -o main main.c -lm
	
test: $(FILES)
	gcc $(CFLAGS) -O3 -ffast-math -o test main.c -lm
	
leak: $(FILES)
	gcc -pedantic -Wall -g -o leak main.c -lm
	valgrind --leak-check=full ./leak
	rm leak
	
.PHONY: clean
clean:
	rm -f main test leak
