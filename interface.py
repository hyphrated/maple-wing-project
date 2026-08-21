import re
from ollama import chat
from model_selection import model_choice, user_input
from model_selection import get_vram_gb, remove_thinking

print(f"Available VRAM: {get_vram_gb():.2f} GB")
print(f"Selected model based on VRAM: {model_choice()}")

model_response = [
    {
        "role": "system",
        "content": "Your name is Fern. Your answers are to be direct and concise, with no unnecessary elaboration."
    },
    {
        "role": "user",
        "content": user_input()
    }
]

def model_process():
    response = chat(
        model=model_choice(),
        messages=model_response,
        think=False
    )
    clean_response = remove_thinking(response.message.content)
    print("Fern:" + clean_response)

model_process()
