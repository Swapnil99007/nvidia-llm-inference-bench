FROM nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12

WORKDIR /app

COPY scripts/run_trtllm_benchmark.py /app/scripts/run_trtllm_benchmark.py
COPY scripts/run_trtllm_benchmark.sh /app/scripts/run_trtllm_benchmark.sh
COPY configs /app/configs
COPY prompts /app/prompts

CMD ["tail", "-f", "/dev/null"]