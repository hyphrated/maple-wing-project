from ollama import chat

response = chat(
  model='qwen3:8b',
  messages=[{'role': 'user', 'content': 'How many letter r are in strawberry?'},
            {'role': 'system', 'content': 'Your name is Fern. Your answers are to be direct and concise, with no unnecessary elaboration.'}],
  think=False,
  stream=False,
)

print('Thinking:\n', response.message.thinking)
print('Answer:\n', response.message.content)