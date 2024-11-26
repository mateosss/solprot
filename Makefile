all:
	eval c++ -O3 -Wall -shared -std=c++11 -fPIC $(shell python3.13 -m pybind11 --includes) bilinterp.cpp -o bilinterp$(shell python3.13-config --extension-suffix)
