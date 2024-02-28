NVCC = nvcc
CUFLAGS  = -O3 -arch=native --default-stream per-thread --use_fast_math
CUFLAGS += --expt-relaxed-constexpr
# CUFLAGS += -Xptxas -warn-double-usage,-warn-spills
# CUFLAGS += -Werror all-warnings
# CUFLAGS += --profile
CFLAGS  = -Wall,-march=native
CUFLAGS += -Xcompiler $(CFLAGS)
# LDFLAGS =
SRC = $(wildcard *.cu)
OBJ = $(SRC:.cu=.o)
DEPEND = $(wildcard *.h)
TARGET = matmul


build: $(TARGET)

$(TARGET): $(OBJ)
	@echo nvcc -o $@ $^
	@$(NVCC) $(CUFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cu $(DEPEND)
	@echo nvcc -c -o $@ $<
	@$(NVCC) $(CUFLAGS) -c -o $@ $<


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
