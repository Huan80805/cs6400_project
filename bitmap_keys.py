# bitmap_keys.py
from typing import Any

def normalize_like_value(val: Any, op: str) -> Any:
    """For LIKE, strip outer % so build-time and query-time match."""
    op_up = op.upper()
    if op_up == "LIKE" and isinstance(val, str):
        if val.startswith("%") and val.endswith("%") and len(val) >= 2:
            return val[1:-1]
    return val

def make_key(col: str, op: str, val: Any) -> str:
    """
    Canonical key for Roaring bitmaps:
      - col: column name
      - op: 'BETWEEN', '=', 'LIKE', etc.
      - val:
          * scalar for '=' / 'LIKE'
          * 2-tuple/list for 'BETWEEN'
    """
    op_up = op.upper()
    val = normalize_like_value(val, op_up)

    if isinstance(val, (list, tuple)):
        # assume BETWEEN with 2 elements
        if len(val) != 2:
            raise ValueError(f"Expected 2-element range for {col} {op_up}, got {val}")
        low, high = val
        # Use repr to keep float formatting consistent
        val_repr = f"{repr(low)}:{repr(high)}"
    else:
        val_repr = repr(val)

    return f"{col}|{op_up}|{val_repr}"
