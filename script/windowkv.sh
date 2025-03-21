max_capacity_prompt=2048 # 2048, 1024, 512.
attn_implementation="eager" # "eager" "flash_attention_2" "sdpa"

CUDA_VISIBLE_DEVICES="0" \
    /the/path/to/python -u \
    /the/path/to/WindowKV/run_demo.py \
    --model_type llama3_windowkv \
    --model_half \
    --use_cache \
    --attn_implementation ${attn_implementation} \
    --max_capacity_prompt ${max_capacity_prompt} \
    --review_window_size "8" \
    --shared_layers "8" \
    --suffix_max "32" \
    --suffix_avg "16" \
    --model_dir "/the/path/to/models_weight/Meta-Llama-3-8B-Instruct/" \
    --bert_model_dir "/the/path/to/models_weight/bert-base-cased" \
    --classifier_dir "/the/path/to/classifier_checkpoint/best_model.pth"


