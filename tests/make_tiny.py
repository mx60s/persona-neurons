"""Builds a tiny random Qwen3 model and five random LoRA adapters in the published adapter format (r, alpha, all seven
linear targets), so the state arithmetic, server and browser can be exercised without downloading Qwen3-8B.

    python tests/make_tiny.py --tok ../work/tok --out ../work/tiny
"""
import argparse
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ADAPTERS = ["o_amp", "o_sup", "n_amp", "n_sup", "ctrl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.tok)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=192, num_hidden_layers=args.layers,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=2048,
                      tie_word_embeddings=False, bos_token_id=151643, eos_token_id=151645)
    base_dir = os.path.join(args.out, "base")
    model = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
    model.save_pretrained(base_dir)
    tok.save_pretrained(base_dir)
    for i, name in enumerate(ADAPTERS):
        torch.manual_seed(100 + i)
        m = Qwen3ForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16)
        pm = get_peft_model(m, LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, target_modules=TARGETS))
        for n, p in pm.named_parameters():
            if "lora_B" in n:  # default init leaves B = 0; give every adapter a real update
                torch.nn.init.normal_(p, std=0.05)
        pm.save_pretrained(os.path.join(args.out, "adapters", name))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
