from typing import Dict, Tuple, Any, Set
from pyroaring import BitMap
import pickle
from bitmap.bitmap_keys import make_key


class RoaringIndex:
    def __init__(self, path: str):
        with open(path, "rb") as f:
            self._bitmaps: Dict[str, BitMap] = pickle.load(f)
        print(f"RoaringIndex loaded: {len(self._bitmaps)} keys")

    def get_ids_for_filter(
        self,
        filter_dict: Dict[str, Tuple[str, Any]],
    ) -> Set[int]:
        """
        Given a dynamic_filter like:

            {"average_rating": ("BETWEEN", (3.5, 3.7))}

        return the set of product_ids that satisfy it, using Roaring bitmaps.

        The key *must* match how build_bitmaps.py and bitmap_keys.make_key()
        normalize (col, op, val).
        """
        if not filter_dict:
            return set()

        # We currently only support a single predicate per query
        col, (op, val) = next(iter(filter_dict.items()))
        key = make_key(col, op, val)

        bm: BitMap | None = self._bitmaps.get(key)
        if bm is None:
            return set()

        # Convert Roaring bitmap to a Python set[int]
        return set(bm)
