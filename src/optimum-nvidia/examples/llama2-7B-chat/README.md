### Build Tensorrt LLM engine
```bash
python3 build.py --model_dir /data/model/Llama-2-7b-chat-hf/ \
                --output_dir /data/model/llama-2-7B-chat-tensorrt/ \
                --tp_size 1 \
                --pp_size 1 \
                --world_size 1 \
                --num_gpus_per_node 1 \
                --dtype float16 \
                --max_batch_size 1 \
                --max_prompt_length 512 \
                --max_new_tokens 1024 \
                --max_beam_width 1
```