# Debian 12 provides GCC 12 / Boost 1.74. Legacy DSP SDKs are checksum-pinned.
FROM debian:bookworm-slim AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates ccache clang cmake curl ffmpeg git \
        hyperfine libbenchmark-dev libboost-filesystem-dev libboost-iostreams-dev \
        libboost-math-dev libboost-serialization-dev libboost-system-dev \
        libjpeg-dev libpng-dev libsndfile1-dev libsuperlu-dev ninja-build pkg-config python3 \
        ruby ruby-dev ruby-bundler valgrind xz-utils zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY environment/dependencies.json /opt/phaselimiter-bootstrap/environment/dependencies.json
COPY script/fetch_build_dependencies.py /opt/phaselimiter-bootstrap/script/fetch_build_dependencies.py
RUN python3 /opt/phaselimiter-bootstrap/script/fetch_build_dependencies.py \
        --platform linux --prefix /opt/phaselimiter --cache-dir /tmp/sdk-downloads \
    && echo /opt/phaselimiter/legacy/usr/lib/x86_64-linux-gnu > /etc/ld.so.conf.d/phaselimiter.conf \
    && ldconfig \
    && rm -rf /tmp/sdk-downloads /opt/phaselimiter-bootstrap

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    CMAKE_PREFIX_PATH=/opt/phaselimiter/legacy/usr \
    IPPROOT=/opt/phaselimiter/opt/intel/oneapi/ipp/2021.9.0 \
    BUNDLE_GEMFILE=/opt/phaselimiter-gems/Gemfile \
    BUNDLE_PATH=/opt/phaselimiter-gems/vendor
COPY environment/Gemfile environment/Gemfile.lock /opt/phaselimiter-gems/
RUN bundle install --jobs 2 --retry 3
WORKDIR /workspace

FROM build AS development
RUN apt-get update \
    && apt-get install -y --no-install-recommends gdb jq procps ripgrep tmux \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://chatgpt.com/codex/install.sh | sh
