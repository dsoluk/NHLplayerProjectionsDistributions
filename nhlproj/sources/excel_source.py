from typing import Optional
import pandas as pd


def safe_read_excel(path: str, preferred_sheet: Optional[str] = None) -> pd.DataFrame:
    """
    Attempts to read the Excel file.
    - Tries the preferred sheet name first (if provided), otherwise falls back to the first sheet.
    - Returns a DataFrame.
    """
    try:
        if preferred_sheet is not None:
            try:
                df = pd.read_excel(path, sheet_name=preferred_sheet, engine="openpyxl")
                print(f"Loaded sheet '{preferred_sheet}' from {path}")
                return df
            except Exception as e:
                print(f"Could not open sheet '{preferred_sheet}': {e}. Falling back to the first sheet...")
        # Fall back to first sheet
        df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
        print(f"Loaded first sheet from {path}")
        return df
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel file at {path}: {e}")
