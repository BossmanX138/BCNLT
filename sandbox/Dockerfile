FROM ubuntu@sha256:2198e4c65076e9cb0b3d695219df667c0e3b0072da64ca3c2648aca17dcecaf5

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        bc \
        ca-certificates \
        cargo \
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

COPY benchmark/go/go.mod benchmark/go/go.sum /opt/bcnlt/go-template/
RUN --mount=type=secret,id=host_ca,required=true \
    cd /opt/bcnlt/go-template \
    && export SSL_CERT_FILE=/run/secrets/host_ca \
    && printf '%s\n' 'package closure' 'import (' '    _ "github.com/go-openapi/inflect"' '    _ "github.com/stretchr/testify/assert"' ')' > closure.go \
    && go mod download \
    && go mod vendor \
    && go mod verify \
    && mkdir -p /opt/bcnlt/go-build-cache \
    && GOCACHE=/opt/bcnlt/go-build-cache go test -mod=vendor -run '^$' . \
    && rm closure.go

COPY benchmark/rust/Cargo.toml benchmark/rust/Cargo.lock /opt/bcnlt/rust-template/
RUN --mount=type=secret,id=host_ca,required=true \
    mkdir -p /opt/bcnlt/rust-template/src /opt/bcnlt/cargo-home \
    && printf '%s\n' '#![allow(dead_code)]' > /opt/bcnlt/rust-template/src/lib.rs \
    && cd /opt/bcnlt/rust-template \
    && export SSL_CERT_FILE=/run/secrets/host_ca \
    && export CARGO_HTTP_CAINFO=/run/secrets/host_ca \
    && CARGO_HOME=/opt/bcnlt/cargo-home cargo fetch --locked \
    && CARGO_HOME=/opt/bcnlt/cargo-home CARGO_NET_OFFLINE=true cargo test --quiet --offline --locked --no-run \
    && rm -rf /opt/bcnlt/rust-template/target/debug/incremental

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
