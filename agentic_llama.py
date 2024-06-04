# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("julianchu/agentic_llama")
model = AutoModelForCausalLM.from_pretrained("julianchu/agentic_llama")
