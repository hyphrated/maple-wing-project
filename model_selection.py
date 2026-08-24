import subprocess

# This section is for automatically choosing a model based 
# on the amount of VRAM available on the GPU.
def get_vram_gb():
    result = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits"
    ])

    vram_mb = int(result.decode().strip().splitlines()[0])
    return vram_mb / 1024
def model_choice():
    vram_amount = get_vram_gb()
    if vram_amount >= 7.9:
        return 'qwen3:8b'
    elif vram_amount >= 6:
        return 'qwen3:4b'
    else:
        return 'qwen3:1.7b' 

# This section is for getting user input and
# handling exit commands.
def user_input():
    message = input("You: ").lower()

    exit_commands = ["exit", "quit", "stop", "end"]
    if message in exit_commands:
        print("Exiting the program.")

    return message

# Cleans up model's response from any thinking tags and returns the cleaned text.
def remove_thinking(text):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    return text.strip()


