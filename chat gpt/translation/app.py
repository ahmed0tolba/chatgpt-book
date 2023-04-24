import nltk # pip install nltk
from transformers import AutoTokenizer, AutoModelForCausalLM  # pip install transformers
import torch

# Load the pre-trained ChatGPT model and tokenizer for English-Arabic translation
model_name = "mofawzy/gpt-chatbot-tiny-english-arabic"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Example text to be translated
english_text = "This is a sample text to be translated from English to Arabic."

# Tokenize the input text
input_ids = tokenizer.encode(english_text, return_tensors="pt")

# Generate the translation using ChatGPT
generated_ids = model.generate(input_ids=input_ids, max_length=50, do_sample=True)

# Decode the generated tokens to get the translated text
arabic_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# Print the translated text
print(arabic_text)