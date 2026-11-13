FROM ubuntu@sha256:2198e4c65076e9cb0b3d695219df667c0e3b0072da64ca3c2648aca17dcecaf5

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        default-jdk-headless \
        g++ \
        gcc \
        golang-go \
        kotlin \
        mono-mcs \
        mono-runtime \
        node-typescript \
        nodejs \
        php-cli \
        python3-minimal \
        ruby \
        rustc \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# The minimal interpreter omits modules used by the bounded JSON protocol.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

COPY runner.py /opt/bcnlt/runner.py
RUN chmod 0555 /opt/bcnlt/runner.py

ENV HOME=/work \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /work
ENTRYPOINT ["python3", "-I", "-B", "/opt/bcnlt/runner.py"]
