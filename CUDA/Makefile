CC:=/usr/local/cuda-11.2/bin/nvcc
#OPT:=-arch=sm_21

main2: cuda2.o main2.o
	${CC} ${OPT} $^ -o $@
#	export LD_LIBRARY_PATH=/home/edu/cuda2/lib

cuda2.o: cuda2.cu
	${CC} ${OPT} -c $^  -o $@

main2.o: main2.cpp
	${CC} ${OPT} -c $^  -o $@

clean:
	rm *.o main2
