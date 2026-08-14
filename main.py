from ollama import chat
from testtime import get_time

result = get_time()

# Eventually organize test time functionality

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

# Added tool calling, currently testing how it works.
    response = chat(
        model='qwen3:8b',
        messages=messages,
        tools = [get_time]
    )

    # Testing tooling calls, work on this later.
    # print("Fern:", response.message.content)
    print(response.message.tool_calls)

    messages.append({
        'role': 'assistant',
        'content': response.message.content
    })

print(response.message.tool_calls)
