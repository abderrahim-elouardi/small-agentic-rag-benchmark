
import logging

import torch

from small_agentic_rag.llm.model_loader import ModelLoader

logging.basicConfig(level=logging.INFO)

loader = ModelLoader(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
)

tokenizer, model = loader.load()

messages = [
    {
        "role": "user",
        "content": "Explique brièvement le rôle d'un système RAG.",
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to(loader.device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
