from small_agentic_rag.llm.model_loader import ModelLoader
from small_agentic_rag.llm.inference import TextGenerator

loader = ModelLoader(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
)

tokenizer, model = loader.load()

generator = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    device=loader.device,
    max_new_tokens=120,
)

prompt = (
    "Question : Que signifie RAG en intelligence artificielle ?\n"
    "Réponds en français en deux phrases.\n"
    "Réponse :"
)

print(generator.generate(prompt, do_sample=False))
