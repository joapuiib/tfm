# Compiler and flags
CXX = g++
CXXFLAGS = `pkg-config --cflags opencv4` -O2 -Wall
LDFLAGS = `pkg-config --libs opencv4`

# Source files and output binary
SRC1 = create_codebook.cpp
SRC2 = transform_image.cpp
TARGET1 = create_codebook
TARGET2 = transform_image

# Header files
HEADERS = kmeans.h

# Default target
all: $(TARGET1) $(TARGET2)

# Compile create_codebook
$(TARGET1): $(SRC1) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SRC1) -o $(TARGET1) $(LDFLAGS)

# Compile transform_image
$(TARGET2): $(SRC2) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SRC2) -o $(TARGET2) $(LDFLAGS)

# Clean up build files
clean:
	rm -f $(TARGET1) $(TARGET2)
