from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Union
import polars as pl  # type: ignore

@dataclass
class DataLoader:
   
    data_path: Path

    def load_data_from_csv(self) -> List[Dict[str, Union[str, int, float, None]]]:
     
        try:
            # Load the CSV file using Polars with robust parameters
            data_frame = pl.read_csv(
                self.data_path,
                infer_schema_length=10000,
                ignore_errors=True,
                encoding="utf8",
                null_values=["NA", ""]
            )
            return data_frame.to_dicts()
        except FileNotFoundError:
            print(f"Error: File not found at {self.data_path}")
            return []
        except pl.exceptions.PolarsError as e:
            print(f"Error reading CSV with Polars: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error loading data from CSV: {e}")
            return []

    def validate_columns(self, required_columns: List[str]) -> bool:
      
        data = self.load_data_from_csv()

        if not data:
            print("Error: No data loaded. Cannot validate columns.")
            return False

        actual_columns = data[0].keys()
        missing_columns = [col for col in required_columns if col not in actual_columns]

        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            return False

        return True
