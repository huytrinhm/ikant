all:
	g++ src/kan/*.cpp src/main.cpp -o main.exe -std=c++20 -Llib -Iinclude -ltinyfiledialogs -lole32 -lcomdlg32 -lraylib -lwinmm -lgdi32 -llapacke -llapack -lblas -lgfortran -Wno-enum-compare