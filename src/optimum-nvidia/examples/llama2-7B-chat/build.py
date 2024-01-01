from transformers import AutoTokenizer
import argparse
from optimum.nvidia import TensorRTEngineBuilder
from optimum.nvidia.models.llama import LlamaWeightAdapter

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    
    # Model topology (sharding, pipelining, dtype)
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--pp_size', type=int, default=1)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--num_gpus_per_node', type=int, default=1)
    parser.add_argument('--dtype', type=str, default='fp16')

    # TensorRT's optimization profiles
    parser.add_argument('--max_batch_size', type=int, default=1)
    parser.add_argument('--max_prompt_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--max_beam_width', type=int, default=1)
    

    args = parser.parse_args()
    return args

def main(args):
    builder = (
            TensorRTEngineBuilder.from_pretrained(args.model_dir, adapter=LlamaWeightAdapter)
            .to(args.dtype)
            .shard(
                tp_degree = args.tp_size,
                pp_degree = args.pp_size,
                world_size = args.world_size,
                num_gpus_per_node = args.num_gpus_per_node
                )
            .with_generation_profile(
                max_batch_size = args.max_batch_size,
                max_prompt_length = args.max_prompt_length,
                max_new_tokens = args.max_new_tokens,
                )
            .with_sampling_strategy(1)
        )

    builder.build(args.output_dir, optimization_level=0)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)