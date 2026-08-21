import subprocess

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
    if vram_amount >= 8:
        return 'qwen3:8b'
    elif vram_amount >= 6:
        return 'qwen3:4b'
    else:
        return 'qwen3:1.7b' 

def user_input():
    message = input("You: ").lower()

    exit_commands = ["exit", "quit", "stop", "end"]
    if message in exit_commands:
        print("Exiting the program.")

    return message


