"""Run every lesson in order, or a few:  python run_all.py   |   python run_all.py 5 6 7"""
import subprocess
import sys
from pathlib import Path

lessons = sorted((Path(__file__).parent / "lessons").glob("[0-9][0-9]_*.py"))
wanted = {int(a) for a in sys.argv[1:] if a.isdigit()}
for path in lessons:
    if wanted and int(path.name[:2]) not in wanted:
        continue
    print(f"\n\n######## {path.name} ########\n", flush=True)
    code = subprocess.call([sys.executable, str(path)])
    if code != 0:
        sys.exit(f"{path.name} failed with exit code {code}")
