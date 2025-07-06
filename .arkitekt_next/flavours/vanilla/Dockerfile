FROM python:3.11-slim-bookworm

# Copy uv binaries
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


# Install git, build tools, and system dependencies (excluding Qt)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        python3.11-dev \
        libssl-dev \
        libffi-dev \
        pkg-config \
        cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy project files and install dependencies via uv
ADD . /app
WORKDIR /app
# Set up uv-managed virtualenv in a known location and sync deps
RUN uv venv /app/.venv && uv sync --locked 
ENV PATH="/app/.venv/bin:$PATH"

