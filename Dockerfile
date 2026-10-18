FROM gcc:16

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        clang \
        cmake \
        gdb \
        git \
        jq \
        libpq-dev \
        ninja-build \
        procps \
        python3 \
        python3-pip \
        ripgrep \
        tmux \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://chatgpt.com/codex/install.sh | sh

WORKDIR /workspace
