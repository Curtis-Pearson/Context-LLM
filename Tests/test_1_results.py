import pandas as pd
import os

def read_csv(file_path: str) -> pd.DataFrame:
    """
    Context:
    This function reads a CSV file and returns its contents as a pandas DataFrame.

    Parameters:
    "file_path": str -> The path to the CSV file to be read.

    Returns:
    "data": pd.DataFrame -> The contents of the CSV file as a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        print("The file does not exist.")
        return None

    try:
        data: pd.DataFrame = pd.read_csv(file_path)
        return data
    except pd.errors.ParserError:
        print("The file is not a valid CSV file.")
        return None

### Test cases
# The file 'salaries.csv' exists and is a valid CSV file.
print(read_csv('salaries.csv'))

# The file 'nonexistent.csv' does not exist.
print(read_csv('nonexistent.csv'))

# The file 'invalid.csv' is not a valid CSV file.
print(read_csv('invalid.csv'))
###

def main(file_path: str):
    """
    Context:
    The purpose of the program is to read and output the contents of a file named 'salaries.csv'.

    Parameters:
    "file_path": str -> The path to the CSV file to be read by the program.

    Returns:
    None
    """
    data: pd.DataFrame = read_csv(file_path)
    
    if data is not None:
        print(data)
    else:
        print("An error occurred while reading the file.")

### Test cases
# The program is run with a valid CSV file.
main('salaries.csv')

# The program is run with a non-existent CSV file.
main('nonexistent.csv')

# The program is run with an invalid CSV file.
main('invalid.csv')
###
