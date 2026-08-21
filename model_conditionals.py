import subprocess

def get_vram_gb():
    result = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits"
    ])

    vram_mb = int(result.decode().strip().splitlines()[0])
    return vram_mb / 1024

print(get_vram_gb())