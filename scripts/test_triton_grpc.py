import queue
import numpy as np
import tritonclient.grpc as grpcclient


def main():
    q = queue.Queue()

    def callback(result, error):
        q.put((result, error))

    client = grpcclient.InferenceServerClient(url="localhost:8001")
    client.start_stream(callback=callback)

    text_input = np.array([["Explain GPU inference benchmarking in 3 short points."]], dtype=object)
    max_tokens = np.array([[64]], dtype=np.int32)

    inputs = []

    inp = grpcclient.InferInput("text_input", text_input.shape, "BYTES")
    inp.set_data_from_numpy(text_input)
    inputs.append(inp)

    inp = grpcclient.InferInput("max_tokens", max_tokens.shape, "INT32")
    inp.set_data_from_numpy(max_tokens)
    inputs.append(inp)

    client.async_stream_infer(
        model_name="ensemble",
        inputs=inputs,
    )

    result, error = q.get(timeout=120)
    client.stop_stream()

    if error:
        raise RuntimeError(error)

    output = result.as_numpy("text_output")
    print(output)


if __name__ == "__main__":
    main()
