"""Write-then-rename so a killed/crashed process can never leave a truncated
or partially-written parquet file at the target path — the target path
only ever has a fully-written temp file renamed onto it (atomic on the
same filesystem), never a file being written in place."""
import os
import tempfile
import pandas as pd


def atomic_to_parquet(df: pd.DataFrame, path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=d or ".", suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
