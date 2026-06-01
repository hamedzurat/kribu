.PHONY: setup build hot-build test test-cpp test-python clean format lint run benchmark view-benchmark view consolidate-dataset generate extract-data tensorboard train arena todo

# Detect number of processors for parallel execution
NPROC := $(shell nproc)

export CC := clang
export CXX := clang++
PREFIX ?= ./dist

setup:
	uv sync
	uv run cmake -B build -S .
	# Link compile_commands.json to root for IDE support (VS Code / CLion)
	ln -sf build/compile_commands.json .
	uv run pre-commit install

build:
	uv run cmake -B build -S .
	uv run cmake --build build -j$(NPROC)
	ccache -s

hot-build:
	@command -v entr >/dev/null 2>&1 || (echo "Error: 'entr' is not installed. Please install it to use hot-build." && exit 1)
	find engine bindings -name "*.cpp" -o -name "*.hpp" | entr -c uv run cmake --build build -j$(NPROC)

test: test-cpp test-python

test-cpp: build
	./build/engine/kribu_engine_tests

test-python:
	PYTHONPATH=python/src uv run pytest python/tests/

run:
	PYTHONPATH=python/src uv run python python/src/kribu/main.py

benchmark: build
	./build/engine/benchmark/kribu_benchmark_main

view-benchmark:
	PYTHONPATH=python/src uv run python3 scripts/view_game.py $(FILE)

generate:
	PYTHONPATH=python/src uv run python python/src/kribu/generate_dataset.py --games 50 --output dataset.parquet --depth 4

extract-data:
	PYTHONPATH=python/src uv run python scripts/create_training_duckdb.py

tensorboard:
	PYTHONPATH=python/src uv run tensorboard --logdir checkpoints/tensorboard

train:
	PYTHONPATH=python/src uv run python -m trainer

# Forward trailing positional arguments to make arena (e.g. make arena -o random)
ifeq (arena,$(firstword $(MAKECMDGOALS)))
  ARENA_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(ARENA_ARGS):;@:)
endif

arena:
	PYTHONPATH=python/src uv run python -m arena $(ARENA_ARGS) $(ARGS)

duckdbui:
	duckdb -cmd "CALL start_ui_server();" benchmark/dataset.duckdb

format:
	PYTHONPATH=python/src uv run ruff format python/
	find engine bindings \( -name "*.cpp" -o -name "*.hpp" -o -name "*.c" -o -name "*.h" -o -name "*.cc" -o -name "*.cxx" -o -name "*.hh" -o -name "*.hxx" \) | xargs -P $(NPROC) uv run clang-format -i
	find . -maxdepth 3 \( -name "CMakeLists.txt" -o -name "*.cmake" \) -not -path "*/build/*" -not -path "*/.venv/*" | xargs -P $(NPROC) uv run cmake-format -i
	find . -maxdepth 2 -name "*.md" -not -path "*/.venv/*" | xargs -P $(NPROC) uv run mdformat

lint:
	PYTHONPATH=python/src uv run ruff check python/
	run-clang-tidy -p build/ -j 8 -quiet -header-filter="engine/.*|bindings/.*" -warnings-as-errors='*' "engine/.*"

todo:
	rg --line-number --color=always -i '\b(TODO|FIXME|BUG|HACK|XXX)\b' --glob '!Makefile' --glob '!doc' || true

clean:
	git clean -Xdf

doc:
	doxygen Doxyfile

doc-view: doc
	uv run python scripts/view_docs.py
