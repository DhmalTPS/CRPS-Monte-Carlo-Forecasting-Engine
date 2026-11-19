# Thin wrapper kept for the documented layout; the real build file is ../Dockerfile.
#   docker build --platform linux/amd64 -f submission/Dockerfile -t agenthon-t2:1.0.0 .
# (build context = repository root)
FROM python:3.13-slim-bookworm
LABEL qfbench2.interface_version="2.0" qfbench2.track="forecasting" qfbench2.verb="forecast"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 PIP_NO_CACHE_DIR=1 HOME=/tmp \
    MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    AGENTHON_CONFIG_DIR=/opt/agenthon/configs AGENTHON_PROMPT_DIR=/opt/agenthon/prompts AGENTHON_PROFILE=final
COPY requirements.txt /opt/agenthon/requirements.txt
RUN pip install --no-cache-dir -r /opt/agenthon/requirements.txt
COPY pyproject.toml README.md /opt/agenthon/
COPY src /opt/agenthon/src
COPY configs /opt/agenthon/configs
COPY prompts /opt/agenthon/prompts
COPY submission/entrypoint.sh /usr/local/bin/agenthon-entrypoint
RUN pip install --no-cache-dir --no-deps /opt/agenthon && chmod +x /usr/local/bin/agenthon-entrypoint \
 && command -v forecast
WORKDIR /tmp
USER 65534:65534
CMD ["forecast", "--help"]
