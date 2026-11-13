FROM swift:6.0.3-jammy@sha256:9ad653a046d5063fff8dec9ae362e25f945bc1965e7dc8773409da71f784c708

COPY runner.py /opt/bcnlt/runner.py
RUN chmod 0555 /opt/bcnlt/runner.py

ENV HOME=/work \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /work
ENTRYPOINT ["python3", "-I", "-B", "/opt/bcnlt/runner.py"]
