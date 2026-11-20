CXX ?= g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall -Wextra -pthread

edgeinfer: edgeinfer.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

run: edgeinfer
	./edgeinfer

clean:
	rm -f edgeinfer
