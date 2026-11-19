# Agenthon 2026 Track 2 submission image (linux/amd64).
#
#   docker build --platform linux/amd64 -t agenthon-t2:1.0.0 .
#   docker run --rm --network=none --read-only --user 65534:65534 \
#     --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m \
#     -v <unit-dir>:/input:ro -v <out>:/output agenthon-t2:1.0.0 \
#     forecast --panels /input/panels/ --text /input/text/ --asof YYYY-MM-DD --out /output/forecast.parquet
#
# Contract points (SUBMISSION_CLI.md, DEVELOPMENT-RUNTIME.md):
#  * the verb `forecast` is an executable on PATH, no ENTRYPOINT (the harness passes the verb);
#  * LABEL qfbench2.interface_version="2.0";
#  * read-only root fs, uid 65534 with no home -> HOME=/tmp, all caches under /tmp;
#  * 256-PID cap -> BLAS/OpenMP pools pinned to 4 threads;
#  * no model weights, no network at build-time dependency on the unit; House calls only via
#    MODEL_ENDPOINT through the injected proxy at run time.
FROM python:3.13-slim-bookworm

LABEL qfbench2.interface_version="2.0" \
      qfbench2.track="forecasting" \
      qfbench2.verb="forecast" \
      org.opencontainers.image.title="agenthon-track2" \
      org.opencontainers.image.version="1.0.1"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp \
    XDG_CACHE_HOME=/tmp \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    AGENTHON_CONFIG_DIR=/opt/agenthon/configs \
    AGENTHON_PROMPT_DIR=/opt/agenthon/prompts \
    AGENTHON_PROFILE=final

COPY requirements.txt /opt/agenthon/requirements.txt
RUN pip install --no-cache-dir -r /opt/agenthon/requirements.txt

COPY pyproject.toml README.md /opt/agenthon/
COPY src /opt/agenthon/src
COPY configs /opt/agenthon/configs
COPY prompts /opt/agenthon/prompts
RUN pip install --no-cache-dir --no-deps /opt/agenthon \
 && python -c "import agenthon, agenthon.main; print('agenthon', agenthon.__version__)" \
 && command -v forecast \
 && python -m compileall -q /usr/local/lib/python3.13/site-packages/agenthon

WORKDIR /tmp
USER 65534:65534
CMD ["forecast", "--help"]
