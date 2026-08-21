from ollama import chat
from tools import get_time
from model_conditionals import get_vram_gb

result = get_time()

messages = [
    {
        'role': 'system',
        'content': 'Your name is Fern. Your answers are to be direct and concise, with no unnecessary elaboration.'
    }
]

while True:
    def thinking_conditional(user_input):
        thinking_words = [
            "why",
            "explain",
            "analyze",
            "compare",
            "debug",
            "solve",
            "reason",
            "calculate",
            "how",
        ]

        # WORKING ON CONDITIONAL THINKING PROMPT. IF USER INPUT CONTAINS ANY OF THE WORDS IN THE LIST, THINKING WILL BE TRUE.

    user_input = input("You: ").lower()


    if user_input.lower() == "exit":
        break

    messages.append({
        'role': 'user',
        'content': user_input
    })

    vram_amount = get_vram_gb()
    model_choice = 'qwen3:4b'

    if vram_amount >= 8:
        model_choice = 'qwen3:8b'
    elif vram_amount >= 4:
        model_choice = 'qwen3:4b'
    print(f"Using model: {model_choice} based on VRAM: {vram_amount} GB")
        
# Added tool calling, currently testing how it works.
    response = chat(
        model=model_choice,
        messages=messages,
        tools = [get_time],
        think=False,
    )

    
    print("Fern:", response.message.content)

    messages.append({
        'role': 'assistant',
        'content': response.message.content
    })

