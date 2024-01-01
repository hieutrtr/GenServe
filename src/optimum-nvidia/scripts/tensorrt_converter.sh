docker run -it --rm --name nvidia-server \
    --gpus all --rm \
    --network triton \
    -v ./third-party/tensorrt-llm/:/root/tensorrt-llm/ \
    -v /datadrive03/cache/llama-2-7B-chat-hf/:/data/model/Llama-2-7b-chat-hf/ \
    -v /datadrive03/cache/llama-2-7B-chat-tensorrt/:/data/model/llama-2-7B-chat-tensorrt/ \
    optimum-tritonserver:latest bash