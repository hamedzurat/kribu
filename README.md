# Kribu

High-performance Sholo Guti AI engine.

![Sholo Guti Banner](asset/shologuti%20banner.avif)

Kribu is a hybrid, high-performance engine and artificial intelligence for **Sholo Guti** (Sixteen Soldiers), a traditional two-player abstract strategy board game popular in Southeast Asia.

![Someone Playing Sholo Guti](asset/PXL_20260617_061815677.jpg)

______________________________________________________________________

## Architecture Overview

Kribu is designed for maximum speed and efficient model training:

- **Core Engine (`engine/`)**: A coordinate-free bitboard representation, compile-time graph topology, and search logic written in pure **C++20**.
- **Bindings (`bindings/`)**: Seamless C++ to Python integration powered by `nanobind`.
- **Python Layer (`python/`)**: Wrapper logic, dataset generation, training pipelines, and the interactive graphical interface.

![How the AI Works](asset/shologuti%20ai%20works.avif)

______________________________________________________________________

## Graphical Interface

Kribu features a graphical interface to play against different AI models or watch them compete.

![Kribu WebUI](asset/2026-06-17--14-47-44-3926.avif)

To launch the graphical interface:

```bash
make gui
```

______________________________________________________________________

## Performance & Model Evaluation

Below is the Elo rating and time-tradeoff log comparing the performance of the various trained models (e.g., minimax, MCTS, and joint policy networks):

![Model Elo Ratings & Time-Tradeoff](asset/elo_time_tradeoff_log.avif)

______________________________________________________________________

## Quick Start

Ensure you have `uv` and `cmake` installed.

```bash
# Setup dependencies and project structure
make setup

# Build the C++ extension
make build

# Run all tests
make test
```

### Developer Commands

- **Release Build**: `make release`
- **Run Simulation**: `make run`
- **Auto-Rebuild on Changes**: `make hot-build`
- **Format & Lint**: `make format && make lint`
