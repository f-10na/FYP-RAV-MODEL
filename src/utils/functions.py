#--- FIND PROJECT ROOT ---
from pathlib import Path

def find_project_root(marker="README.md"):
    path = Path.cwd().resolve()
    for parent in [path] + list(path.parents):
        if (parent / marker).exists():
            return parent
    raise RuntimeError("Project root not found")

PROJECT_ROOT = find_project_root()
print(f"Project root found at: {PROJECT_ROOT}")

#--- LOAD CSV UTILITY FUNCTION ---
import pandas as pd
def load_csv(filepath):
    return pd.read_csv(
        filepath,
        encoding="utf-8",
        low_memory=False
    )
