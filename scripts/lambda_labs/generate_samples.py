#!/usr/bin/env python3
"""
Generate text samples from trained HC and mHC models.

Run AFTER main experiments complete. This loads checkpoints and generates
sample text to show qualitative differences between HC and mHC outputs.

Usage:
    python scripts/lambda_labs/generate_samples.py

Output:
    - Prints samples to console
    - Saves to samples.txt
"""

import os
import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.lambda_labs.train_c4 import GPT, ModelConfig

import tiktoken


def load_model(run_dir: Path, device: str = "cuda") -> tuple:
    """Load model from checkpoint."""
    config_path = run_dir / "config.json"
    model_path = run_dir / "final_model.pt"

    if not config_path.exists() or not model_path.exists():
        return None, None

    with open(config_path) as f:
        config_dict = json.load(f)

    model_config = ModelConfig(**config_dict["model"])
    model = GPT(model_config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, model_config


@torch.no_grad()
def generate(
    model: GPT,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    device: str = "cuda",
) -> str:
    """Generate text from prompt."""
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        # Crop to max sequence length
        tokens_cond = tokens[:, -model.config.max_seq_len:]

        # Forward pass
        logits, _ = model(tokens_cond)
        logits = logits[:, -1, :] / temperature

        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_token], dim=1)

        # Stop at end of text
        if next_token.item() == tokenizer.eot_token:
            break

    return tokenizer.decode(tokens[0].tolist())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = tiktoken.get_encoding("gpt2")

    # Prompts to test
    prompts = [
        "The future of artificial intelligence is",
        "In a shocking discovery, scientists found that",
        "The best way to learn programming is to",
    ]

    # Find models to compare
    runs_dir = Path("runs_c4")
    models_to_compare = [
        ("hc_d32_s42", "HC (depth 32)"),
        ("mhc_d32_s42", "mHC (depth 32)"),
    ]

    output_lines = []
    output_lines.append("=" * 70)
    output_lines.append("TEXT GENERATION COMPARISON: HC vs mHC")
    output_lines.append("=" * 70)

    for prompt in prompts:
        output_lines.append(f"\nPrompt: \"{prompt}\"\n")
        output_lines.append("-" * 50)

        for run_name, label in models_to_compare:
            run_dir = runs_dir / run_name
            model, config = load_model(run_dir, device)

            if model is None:
                output_lines.append(f"\n{label}: [Model not found at {run_dir}]")
                continue

            try:
                generated = generate(model, tokenizer, prompt, max_tokens=100, device=device)
                output_lines.append(f"\n{label}:")
                output_lines.append(generated)
            except Exception as e:
                output_lines.append(f"\n{label}: [Generation failed: {e}]")

            # Free memory
            del model
            torch.cuda.empty_cache()

        output_lines.append("")

    output_lines.append("=" * 70)

    # Print and save
    output_text = "\n".join(output_lines)
    print(output_text)

    with open("samples.txt", "w") as f:
        f.write(output_text)

    print(f"\nSaved to samples.txt")


if __name__ == "__main__":
    main()
