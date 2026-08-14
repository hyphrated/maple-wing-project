from ollama import chat

messages = [
    {
        'role': 'system',
        'content': 'Your name is Fern. Your answers are to be direct and concise, with no unnecessary elaboration.'
    }
]

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    messages.append({
        'role': 'user',
        'content': user_input
    })

    response = chat(
        model='qwen3:8b',
        messages=messages
    )

    print("Fern:", response.message.content)

    messages.append({
        'role': 'assistant',
        'content': response.message.content
    })

