"""Use the working GPU Python. Never install or replace PyTorch."""
import os
import subprocess
import sys
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    check = "import torch; assert torch.cuda.is_available(), 'Select your existing CUDA Python/kernel'; print('GPU:', torch.cuda.get_device_name(0)); print('PyTorch:', torch.__version__)"
    subprocess.run([sys.executable, '-c', check], check=True)
    env = os.environ.copy()
    # Older notebooks set a constraint to a CUDA runtime version that differs
    # from installed package metadata. Nothing in our requirements requests torch.
    env.pop('PIP_CONSTRAINT', None)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(root/'requirements.txt')], env=env, check=True)
    subprocess.run([sys.executable, '-m', 'ipykernel', 'install', '--user', '--name', 'boundary-interventions', '--display-name', 'Boundary interventions (GPU)'], check=True)
    subprocess.run([sys.executable, '-c', check], check=True)
    print('Setup complete. PyTorch was not requested by the installer.')

if __name__ == '__main__':
    main()
