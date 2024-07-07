import torch
import urllib3
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
#os.environ["tf_gpu_allocator"]="cuda_malloc_async"

# Load pre-trained GPT-J model and tokenizer
#model_name = 'EleutherAI/gpt-j-6B'
model_name = '/home/saul/.cache/huggingface/hub/models--EleutherAI--gpt-j-6B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
#model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-j-6B' ,from_tf=True)

def generate_text(prompt, max_length=5):
    # Tokenize input prompt
    inputs = tokenizer(prompt, return_tensors='pt')

    # Generate text
    outputs = model.generate(
        inputs.input_ids, 
        max_length=max_length, 
        num_return_sequences=1, 
        no_repeat_ngram_size=1, 
        early_stopping=True
    )

    # Decode the generated text
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Define the context or scenario for causal inference
context = """
You are an expert in causal inference. Given the following scenario, explain the possible causal relationships:
"Studies have shown that increased physical activity leads to improved mental health. Furthermore, it has been observed that individuals who engage in regular exercise tend to have lower stress levels and better sleep quality."
"""

# Construct the prompt for the LLM
prompt = f"{context}\n\nPlease describe the causal relationships in detail, considering possible confounding variables and mechanisms:"
#prompt = context + "  Please describe the causal relationships in detail, considering possible confounding variables and mechanisms:"

# Get the causal inference from the LLM
causal_inference = generate_text(prompt)

# Print the result
print("Causal Inference Explanation:\n", causal_inference)
