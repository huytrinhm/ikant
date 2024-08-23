all:
	g++ src/main.cpp -o main.exe -std=c++20 -Llib -Iinclude -lraylib -lwinmm -lgdi32 -Wno-enum-compare