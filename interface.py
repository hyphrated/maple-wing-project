from ollama import chat
from model import model_choice, user_input
from model import get_vram_gb

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

def mode_process():
    response = chat(
        model=model_choice(),
        messages=model_response,
        think=False
    )

    print("Fern:" + response.message.content)

