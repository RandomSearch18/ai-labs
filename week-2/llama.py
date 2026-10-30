import torch
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_31 = "meta-llama/Llama-3.1-8B-Instruct"  # <-- llama 3.1
llama_32 = "meta-llama/Llama-3.2-3B-Instruct"  # <-- llama 3.2

prompt = [
    {
        "role": "system",
        "content": "You are a helpful assistant, that responds as a pirate.",
    },
    {"role": "user", "content": "What's Deep Learning?"},
]

generator = pipeline(model=llama_32, device=device, torch_dtype=torch.bfloat16)
generation = generator(
    prompt, do_sample=False, temperature=1.0, top_p=1, max_new_tokens=50
)

print(f"Generation: {generation[0]['generated_text']}")
