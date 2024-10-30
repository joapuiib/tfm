# Compiler and flags
CXX = g++
CXXFLAGS = `pkg-config --cflags opencv4` -O2 -Wall
LDFLAGS = `pkg-config --libs opencv4`

# Source files and output binary
SRC = create_codebook.cpp
TARGET = create_codebook

# Default target
all: $(TARGET)

# Compile the target binary
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

# Clean up build files
clean:
	rm -f $(TARGET)
