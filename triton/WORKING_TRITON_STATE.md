# Working Triton + TensorRT-LLM State

## Status
- Triton launches successfully.
- ensemble model ready endpoint returns HTTP 200.
- tensorrt_llm model ready endpoint returns HTTP 200.
- gRPC inference works through ensemble.

## Working engine build command

trtllm-build \
  --checkpoint_dir /workspace/trt_checkpoints/qwen25_7b \
  --output_dir /workspace/trt_engines/qwen25_7b \
  --gemm_plugin bfloat16 \
  --gpt_attention_plugin bfloat16 \
  --max_batch_size 16 \
  --max_input_len 1024 \
  --max_seq_len 1152 \
  --max_num_tokens 1024 \
  --remove_input_padding enable \
  --paged_kv_cache enable \
  --context_fmha disable

## Critical Triton config fixes
- max_batch_size changed from 64 to 16.
- preferred_batch_size changed from [64] to [16].
- encoder_model_path must be empty for Qwen2.5-7B because it is decoder-only.
- gpt_model_path points to /workspace/trt_engines/qwen25_7b.

## Launch command

tritonserver \
  --model-repository=/workspace/model_repository \
  --log-verbose=1

## Test command

python /app/scripts/test_triton_grpc.py
