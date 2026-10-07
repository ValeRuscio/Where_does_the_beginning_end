"""Launch/resume the main run independently of the notebook connection."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent

def main():
    result=ROOT/'results';result.mkdir(exist_ok=True)
    status_path=result/'job-status.json'
    if status_path.exists():
        status=json.loads(status_path.read_text())
        if status.get('status')=='running' and status.get('pid'):
            try:os.kill(status['pid'],0)
            except ProcessLookupError:pass
            else:
                print(f"A collection process is already running (PID {status['pid']}). See results/main.log.")
                return
    log_path=result/'main.log'
    with log_path.open('a',buffering=1) as log:
        process=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_experiments.py'),'run','--stage','main'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    (result/'background-pid.txt').write_text(str(process.pid))
    time.sleep(.5)
    if process.poll() not in (None,0):
        print('\n'.join(log_path.read_text(errors='replace').splitlines()[-40:]))
        raise RuntimeError('Run failed at startup; see results/main.log')
    print(f'Main run launched/resumed. PID: {process.pid}\nLog: {log_path}\nKeep the GPU instance running. The browser/notebook may disconnect.\nWhen complete: results/main-intervention-results.zip')

if __name__=='__main__':main()
