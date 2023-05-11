CXX 			:= g++
SRC 			:= $(wildcard ./src/*.cpp)
OBJ				:= $(patsubst %.cpp, %.o, $(SRC))
CXXFLAGS 		:= -std=c++11 -pipe -g
SHARED_LIBRARY 	=

# Add needed header path
# SHARED_LIBRARY += -I/usr/local/cuda-11.4/include

# Add dynamic library
SHARED_LIBRARY += -pthread
# SHARED_LIBRARY += -lonnxruntime
# SHARED_LIBRARY += -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn
# SHARED_LIBRARY += -L/usr/local/cuda/lib64 -lcuda -lcudart


GPGPU: update $(OBJ)
	@echo Compiling GPGPU
	@$(CXX) $(CXXFLAGS) -o GPGPU $(OBJ) $(SHARED_LIBRARY)


%.o: %.cpp
	@echo Build $@
	@@$(CXX) $(CXXFLAGS) -c -o $@ $<
#@@$(CXX) $(CXXFLAGS) -MMD -MP -c -o $@ $<

.PHONY: update clean
update:
	@:

clean:
	@find -name "*.o" -exec rm {} \;
#@find -name "*.d" -exec rm {} \;


