from dataclasses import dataclass
from typing import Dict, List, Any
import re


@dataclass
class Cleaner:
    data: List[Dict[str, Any]]

    def rename_with_best_practices(self) -> None:
        if not self.data:
            raise ValueError("Error: No data to rename. Data is empty.")

        original_columns = self.data[0].keys()

        def snake_case(name: str) -> str:
            name = re.sub(r"[^\w\s]", "", name)  
            name = re.sub(r"\s+", "_", name)  
            name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)  
            return name.lower()

        renamed_columns = {col: snake_case(col) for col in original_columns}
        #print(f"Renamed columns: {renamed_columns}")

        self.data = [
            {renamed_columns.get(col, col): value for col, value in (row or {}).items()}
            for row in self.data if row is not None
        ]

    def na_to_none(self) -> List[Dict[str, Any]]:
        if not self.data:
            raise ValueError("Error: No data to clean. Data is empty.")


        clean_data = [
            {key: None if value == "NA" else value for key, value in (row or {}).items()}
            for row in self.data if row is not None
        ]

        return clean_data

    def validate_snake_case_columns(self) -> bool:
        if not self.data:
            raise ValueError("Error: No data to validate. Data is empty.")

        columns = self.data[0].keys()

        non_snake_case = [col for col in columns if not re.match(r"^[a-z0-9_]+$", col)]

        if non_snake_case:
            print(f"Test failed: Column names should be in snake_case: {non_snake_case}")
            return False

        return True
