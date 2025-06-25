FROM ubuntu:24.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y bash \
    libopenmpi-dev \
    build-essential \
    software-properties-common \
    curl \
    ca-certificates \
    procps \
    wget \
    less \
    vim \
    git \
    python3-pip \
    python3-venv \
    python3-setuptools \
    python3-wheel \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN uv python install 3.12
RUN uv venv
RUN uv pip install setuptools wheel torch
RUN uv lock
RUN uv sync
RUN git submodule update --remote --init --recursive --merge

CMD ["/bin/bash"]
