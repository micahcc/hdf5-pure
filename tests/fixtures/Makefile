# Build and run the HDF5 fixture generator.
#
# For deterministic output (fixed timestamps), install libfaketime:
#   brew install libfaketime
#
# Usage:
#   make          # compile + generate fixtures
#   make compile  # compile only
#   make generate # generate fixtures (recompile if needed)
#   make clean    # remove compiled binary

CC = h5cc
BINARY = gen_fixtures
SOURCE = gen_fixtures.c
FAKE_TIME = 2024-01-01 00:00:00

.PHONY: all compile generate clean

all: generate

compile: $(BINARY)

$(BINARY): $(SOURCE)
	$(CC) -o $@ $<

generate: $(BINARY)
	@if command -v faketime >/dev/null 2>&1; then \
		echo "Using faketime for deterministic timestamps"; \
		faketime '$(FAKE_TIME)' ./$(BINARY); \
	else \
		echo "Warning: faketime not found. Timestamps will vary between runs."; \
		echo "Install with: brew install libfaketime"; \
		./$(BINARY); \
	fi

clean:
	rm -f $(BINARY)
