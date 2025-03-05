import transformers
from transformers import (   
            AutoTokenizer,
            LlamaConfig
        )

def get_model_class(model_type):

    if model_type == "llama3_fullkv":
        from models.modeling_llama import LlamaForCausalLM
        config_class = LlamaConfig
        tokenizer_class = AutoTokenizer
        model_class = LlamaForCausalLM

        
    elif model_type == "llama3_windowkv":
        from models.modeling_llama_windowkv import llama_attn_forward_windowkv, llama_flash_attn2_forward_windowkv, llama_sdpa_attn_forward_windowkv
        transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_attn_forward_windowkv
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = llama_flash_attn2_forward_windowkv
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = llama_sdpa_attn_forward_windowkv
        
        from transformers import LlamaForCausalLM
        config_class = LlamaConfig
        tokenizer_class = AutoTokenizer
        model_class = LlamaForCausalLM


        from models.modeling_llama_windowkv import prepare_inputs_for_generation_llama
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_llama

    else:
        raise ValueError(f"Unknown model type {model_type}.")

    return config_class, tokenizer_class, model_class


