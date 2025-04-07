import constants as C
import pandas as pd
import os

from logging_cfg import get_logger
l = get_logger(__name__)

# Path to the meta directory
META_DIR = os.path.join(C.PROJECT_ROOT, "ds", "meta")

def validate_dataframe(df: pd.DataFrame, schema: dict) -> bool:
    """
    Validates if a DataFrame follows the given schema.
    
    :param df: The DataFrame to validate
    :param schema: A dictionary defining the expected schema
    :return: True if valid, False otherwise
    """
    # Check if all required columns exist
    missing_columns = [col for col in schema if col not in df.columns]
    if missing_columns:
        l.warning(f"Missing columns: {missing_columns}")
        return False

    # Check if all columns have the correct data type
    for col, expected_dtype in schema.items():
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            l.warning(f"Column {col} has incorrect type: expected {expected_dtype}, got {actual_dtype}")
            return False

    l.warning("DataFrame is valid")
    return True


def write_csv_meta(df: pd.DataFrame, file_name: str) -> str:
    """
    Write df to csv file in the constant meta directory.
    """
    # TODO: fix the validation lata
    # if validate_dataframe(df, PD_SCHEMA) == False:
    #     raise ValueError("Dataframe validation failed")

    # Ensure the meta directory exists
    os.makedirs(META_DIR, exist_ok=True)

    # Write the dataframe to the meta directory
    file_path = os.path.join(META_DIR, file_name + ".csv")
    df.to_csv(file_path, index=False)
    
    return file_path

def read_csv_as_dataframe(file_path: str) -> pd.DataFrame:
    """
    Read a csv file from the constant meta directory 
    and return it as `pd.DataFrame`.
    """
    # Ensure the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")

    # Ensure the file is a CSV
    if not file_path.endswith('.csv'):
        raise ValueError(f"File is not a CSV: '{file_path}'")
    return pd.read_csv(file_path)


def get_classes_from_config():
    """
    Get class names from the config file.
    """
    from constants import CLASSNAMES_CSV
    with open(CLASSNAMES_CSV, "r") as f:
        df = pd.read_csv(f)
        return df['class_name'].tolist()


def get_classes_ordinal_from_config():
    """
    Get class names from the config file.
    """
    from constants import CLASSNAMES_CSV
    with open(CLASSNAMES_CSV, "r") as f:
        df = pd.read_csv(f)
        return df['id'].tolist()


def main():
    """
    Main function to test get_classes_from_config.
    """
    try:
        class_names = get_classes_from_config()
        print("Class names:", class_names)
    except Exception as e:
        l.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()