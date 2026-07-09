import io
import json
from datetime import datetime
from typing import Optional

import pandas as pd


def export_csv(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def export_json(data: dict | list, indent: int = 2) -> bytes:
    return json.dumps(data, indent=indent, ensure_ascii=False, default=str).encode("utf-8")


def make_export_filename(prefix: str, ext: str, source_name: str = "") -> str:
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    clean_source = source_name.replace(" ", "_").replace(".", "_") if source_name else ""
    parts = [p for p in [prefix, clean_source, date_str] if p]
    return f"{'_'.join(parts)}.{ext}"
