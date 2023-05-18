CXX 			:= g++
SRC 			:= $(filter-out ./src/running.cpp, $(wildcard ./src/*.cpp))
OBJ				:= $(patsubst %.cpp, %.o, $(SRC))
CXXFLAGS 		:= -std=c++17 -pipe -g -O3
SHARED_LIBRARY 	:= -pthread

GPGPU: $(OBJ)
	@echo Compiling GPGPU
	@$(CXX) $(CXXFLAGS) -o GPGPU $(OBJ) $(SHARED_LIBRARY)

RUN:
	@echo Compiling RUN
	@$(CXX) $(CXXFLAGS) -o RUN ./src/running.cpp $(SHARED_LIBRARY)

%.o: %.cpp
	@echo Build $@
	@@$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: update clean RUN
update:
	@:

clean:
	@find -name "*.o" -exec rm {} \;

