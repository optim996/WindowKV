import argparse

def parse_args_func():
    parser = argparse.ArgumentParser(description="demo params config")

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--use_cache",
        action='store_true',
    )

    parser.add_argument(
        "--model_half",
        action='store_true',
    )

    parser.add_argument(
        "--attn_implementation",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--layer_decay_strategy",
        type=str,
        default="arithmetic",
    )

    parser.add_argument(
        "--max_capacity_prompt",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--beta",
        type=int,
        default=14
    )

    parser.add_argument(
        "--review_window_size",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--suffix_max",
        type=int,
        required=True
    )

    parser.add_argument(
        "--suffix_avg",
        type=int,
        required=True
    )

    parser.add_argument(
        "--shared_layers",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--bert_model_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--classifier_dir",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    
    return args


