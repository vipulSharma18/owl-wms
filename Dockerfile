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
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN uv venv
RUN uv pip install -r requirements.txt
RUN uv pip install -U --prerelease allow torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

RUN git submodule update --remote --init --recursive --merge

CMD ["/bin/bash"]