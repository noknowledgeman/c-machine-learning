CFLAGS =  -pedantic -Wall -g
# All the SRCS with implementationn
SRCS = main.c arena.h neuralnetwork.h matrix.h

run: build-main
	./main

test: build-test
	./test
	
build-main: $(SRCS)
	gcc $(CFLAGS) -fsanitize=address -o main main.c -lm
	
build-test: $(SRCS)
	gcc $(CFLAGS) -O3 -ffast-math -o test main.c -lm
	
leak: $(SRCS)
	gcc -pedantic -Wall -g -o leak main.c -lm
	valgrind --leak-check=full ./leak
	rm leak
	
.PHONY: clean leak build-test build-main main test
clean:
	rm -f main test leak
