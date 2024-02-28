CC = nvcc
# CFLAGS = -g -Wall
# LDFLAGS =
SRC = $(wildcard *.cu)
OBJ = $(SRC:.cu=.o)
TARGET = saxpy


build: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(CC) $(CFLAGS) -c -o $@ $<


.PHONY: run
run: build
	@echo
	@./$(TARGET)

.PHONY: profile
profile: build
	nsys profile --stats=true $(TARGET)

.PHONY: clean
clean:
	rm -f $(OBJ) $(TARGET) *~
