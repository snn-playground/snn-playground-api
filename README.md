# Kaptios SNN Visualizer API

## Installation

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd kmc-api
```

2. Install dependencies:

```bash
uv sync
```

This creates a virtual environment and installs all required packages from `pyproject.toml`.

## Development

Run in development mode with auto-reload:

```bash
uv run --active python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://localhost:8000`

## API Documentation

Once the server is running, visit:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Health Check

Check the API status:

```bash
curl http://localhost:8000/health
```
