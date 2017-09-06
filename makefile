all : main.cpp
	g++-7 -lstdc++ -fopenmp main.cpp -o program
	./program
