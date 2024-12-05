from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

@dataclass
class Descriptor:
    data: List[Dict[str, Any]]

    def none_ratio(self, columns: Optional[List[str]] = None) -> Dict[str, float]:
        columns = columns if columns else {key for row in self.data for key in row.keys()}
        result = {}
        for col in columns:
            total_count = len(self.data)
            none_count = sum(1 for row in self.data if row.get(col) is None)
            ratio = none_count / total_count if total_count > 0 else 0
            result[col] = ratio
        return result

    def average(self, columns: Optional[List[str]] = None) -> Dict[str, Optional[float]]:
        columns = columns if columns else {key for row in self.data for key in row.keys()}
        result = {}
        for col in columns:
            values = [row[col] for row in self.data if col in row and isinstance(row[col], (int, float))]
            result[col] = sum(values) / len(values) if values else None
        return result

    def median(self, columns: Optional[List[str]] = None) -> Dict[str, Optional[float]]:
        columns = columns if columns else {key for row in self.data for key in row.keys()}
        result = {}
        for col in columns:
            values = [row[col] for row in self.data if col in row and isinstance(row[col], (int, float))]
            if not values:
                result[col] = None
                continue
            values.sort()
            mid = len(values) // 2
            result[col] = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
        return result

    def percentile(self, columns: Optional[List[str]] = None, p: float = 50,) -> Dict[str, Optional[float]]:
        if not isinstance(p, (int, float)) or not (0 <= p <= 100):
            raise ValueError("p must be a numeric value between 0 and 100.")
        columns = columns if columns else {key for row in self.data for key in row.keys()}
        result = {}
        for col in columns:
            values = [row[col] for row in self.data if col in row and isinstance(row[col], (int, float))]
            if not values:
                result[col] = None
                continue
            values.sort()
            k = (len(values) - 1) * (p / 100)
            f = int(k)
            c = k - f
            result[col] = values[f] * (1 - c) + values[f + 1] * c if f + 1 < len(values) else values[f]
        return result

    def type_and_mode(self, columns: Optional[List[str]] = None) -> Dict[str, Tuple[Optional[str], Optional[Any]]]:
        columns = columns if columns else {key for row in self.data for key in row.keys()}
        result = {}
        for col in columns:
            values = [row[col] for row in self.data if col in row and row[col] is not None]
            if not values:
                result[col] = (None, None)
                continue
            if not all(isinstance(v, type(values[0])) for v in values):
                result[col] = (None, None)
                continue
            mode = max(set(values), key=values.count)
            result[col] = (type(mode).__name__, mode)
        return result


@dataclass
class DescriptorNumpy:
    data: np.ndarray

    def __init__(self, data: List[Dict[str, Any]]):
        keys = {key for row in data for key in row.keys()}
        dtype = [(key, 'O') for key in keys]
        self.data = np.array([tuple(row.get(key, None) for key in keys) for row in data], dtype=dtype)

    def none_ratio(self, columns: Optional[List[str]] = None) -> Dict[str, float]:
        columns = columns if columns else self.data.dtype.names
        return {col: np.mean([x is None for x in self.data[col]]) for col in columns}

    def average(self, columns: Optional[List[str]] = None) -> Dict[str, Optional[float]]:
        columns = columns if columns else self.data.dtype.names
        result = {}
        for col in columns:
            numeric_vals = [x for x in self.data[col] if isinstance(x, (int, float))]
            result[col] = np.mean(numeric_vals) if numeric_vals else None
        return result

    def median(self, columns: Optional[List[str]] = None) -> Dict[str, Optional[float]]:
        columns = columns if columns else self.data.dtype.names
        result = {}
        for col in columns:
            numeric_vals = [x for x in self.data[col] if isinstance(x, (int, float))]
            result[col] = np.median(numeric_vals) if numeric_vals else None
        return result

    def percentile(self, columns: Optional[List[str]] = None, p: float = 50) -> Dict[str, Optional[float]]:
        if not isinstance(p, (int, float)) or not 0 <= p <= 100:
            raise ValueError("p must be a numeric value between 0 and 100.")
        columns = columns if columns else self.data.dtype.names
        result = {}
        for col in columns:
            numeric_vals = [x for x in self.data[col] if isinstance(x, (int, float))]
            result[col] = np.percentile(numeric_vals, p) if numeric_vals else None
        return result

    def type_and_mode(self, columns: Optional[List[str]] = None) -> Dict[str, Tuple[Optional[str], Optional[Any]]]:
        columns = columns if columns else self.data.dtype.names
        result = {}
        for col in columns:
            values = [x for x in self.data[col] if x is not None]
            if not values:
                result[col] = (None, None)
                continue
            if not all(isinstance(v, type(values[0])) for v in values):
                result[col] = (None, None)
                continue
            mode = max(set(values), key=values.count)
            result[col] = (type(mode).__name__, mode)
        return result
