from ollama import chat
from model_selection import model_choice, user_input
from model_selection import get_vram_gb, remove_thinking

print(f"Available VRAM: {get_vram_gb():.2f} GB")
print(f"Selected model based on VRAM: {model_choice()}")

conversation = [
    {
        "role": "system",
        "content": "Your name is Fern. Your answers are to be direct and concise, with no unnecessary elaboration."
    }
]

def model_process():
    selected_model = model_choice()

    while True:
        message = user_input()
        if message in {"exit", "quit", "stop", "end"}:
            break

        conversation.append({"role": "user", "content": message})

        response = chat(
            model=selected_model,
            messages=conversation,
            think=False
        )
        clean_response = remove_thinking(response.message.content)
        conversation.append({"role": "assistant", "content": clean_response})
        print(f"Fern: {clean_response}")

model_process()
