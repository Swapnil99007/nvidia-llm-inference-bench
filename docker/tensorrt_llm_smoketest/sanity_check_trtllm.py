from tensorrt_llm import LLM, SamplingParams


def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    for output in llm.generate(prompts, sampling_params):
        print(
            f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
        )


if __name__ == "__main__":
    main()