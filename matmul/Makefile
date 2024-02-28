NVCC = nvcc
CFLAGS  = -Wall,-march=native
CUFLAGS  = -O3 -arch=native --default-stream per-thread --use_fast_math -Xcompiler $(CFLAGS)
CUFLAGS += -Xptxas -warn-double-usage,-warn-spills
# CUFLAGS += --expt-relaxed-constexpr
# CUFLAGS += -Werror all-warnings
# CUFLAGS += --profile
# LDFLAGS =
SRC = $(wildcard *.cu)
OBJ = $(SRC:.cu=.o)
DEPEND = $(wildcard *.h)
TARGET = matmul


build: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(CUFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cu $(DEPEND)
	$(NVCC) $(CUFLAGS) -c -o $@ $<


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
