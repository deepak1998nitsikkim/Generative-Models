Building a simple chatbot with transformers

1. Transformer Facebook Model - "facebook/blenderbot-400M-distill"

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Selecting the model. You will be using "facebook/blenderbot-400M-distill" in this example.
model_name = "facebook/blenderbot-400M-distill"

# Load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the chat function
def chat_with_bot():
    while True:
        # Get user input
        input_text = input("You: ")

        # Exit conditions
        if input_text.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        # Tokenize input and generate response
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(inputs, max_new_tokens=150) 
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Display bot's response
        print("Chatbot:", response)

# Start chatting
chat_with_bot()


2. Transformer Google Model - "google/flan-t5-base"

import sentencepiece
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

### Let's chat with another bot
def chat_with_another_bot():
    while True:
        # Get user input
        input_text = input("You: ")

        # Exit conditions
        if input_text.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        # Tokenize input and generate response
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        outputs = model.generate(inputs, max_new_tokens=150) 
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Display bot's response
        print("Chatbot:", response)

# Start chatting
chat_with_another_bot()



3. Transformer Google Model - "google/gemma-2b-it"

from transformers import pipeline

chatbot = pipeline(
    "text-generation",
    model="google/gemma-2b-it"
)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    response = chatbot(user_input, max_length=200)
    print("Bot:", response[0]["generated_text"])

4. Transformer HuggingFace Model - "HuggingFaceH4/zephyr-7b-beta"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "HuggingFaceH4/zephyr-7b-beta"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def chat():
    print("Chatbot started! Type 'exit' to stop.")
    messages = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        messages.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        bot_reply = response.split("assistant")[-1].strip()

        print("Bot:", bot_reply)

        messages.append({"role": "assistant", "content": bot_reply})

chat()
