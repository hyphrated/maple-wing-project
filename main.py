from ollama import chat
from tools import get_time

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

    user_input: user_input.lower()
    user_input: input("You: ")


    if user_input.lower() == "exit":
        break

    messages.append({
        'role': 'user',
        'content': user_input
    })

# Added tool calling, currently testing how it works.
    response = chat(
        model='qwen3:8b',
        messages=messages,
        tools = [get_time],
        think=False,
    )

    
    print("Fern:", response.message.content)

    messages.append({
        'role': 'assistant',
        'content': response.message.content
    })

