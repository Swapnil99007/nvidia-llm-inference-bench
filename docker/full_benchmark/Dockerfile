FROM nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12

WORKDIR /app

# Copy full project assets
COPY configs /app/configs
COPY prompts /app/prompts
COPY scripts /app/scripts

# Optional: copy README or docs if you want them inside the container
# COPY README.md /app/README.md

# Install extra Python packages needed by your scripts
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
        pandas \
        matplotlib \
        pyyaml \
        requests \
        transformers \
        accelerate \
        sentencepiece \
        vllm

CMD ["tail", "-f", "/dev/null"]