import pandas as pd

def read_csv(file_path: str) -> pd.DataFrame:
    """
    Context:
    This function is used to read the CSV file.

    Parameters:
    "file_path": str -> The path to the CSV file.

    Returns:
    "dataframe": pd.DataFrame -> The dataframe containing the data from the CSV file.
    """
    try:
        dataframe: pd.DataFrame = pd.read_csv(file_path)
        return dataframe
    except FileNotFoundError:
        print("The file does not exist.")
        return None

### Test cases
# The CSV file exists and is readable.
print(read_csv('salaries.csv'))

# The CSV file does not exist.
print(read_csv('nonexistent.csv'))
###

def get_employees_above_avg_salary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Context:
    This function is used to get the first and last names of employees whose average salary between
    2018-2020 is above £25000.

    Parameters:
    "dataframe": pd.DataFrame -> The dataframe containing the data from the CSV file.

    Returns:
    "employees": pd.DataFrame -> The dataframe containing the first and last names of employees whose
    average salary between 2018-2020 is above £25000.
    """
    # Calculate the average salary of each employee between 2018-2020.
    dataframe['average_salary'] = dataframe[['2018', '2019', '2020']].mean(axis=1)

    # Filter the dataframe to include only employees whose average salary is above £25000.
    employees = dataframe[dataframe['average_salary'] > 25000][['first_name', 'last_name']]

    return employees

### Test cases
# The dataframe contains employees with average salaries above £25000 between 2018-2020.
print(get_employees_above_avg_salary(pd.DataFrame({
    'first_name': ['John', 'Jane', 'Joe'],
    'last_name': ['Doe', 'Doe', 'Bloggs'],
    '2018': [30000, 20000, 26000],
    '2019': [32000, 21000, 27000],
    '2020': [33000, 22000, 28000]
})))

# The dataframe does not contain any employees with average salaries above £25000 between 2018-2020.
print(get_employees_above_avg_salary(pd.DataFrame({
    'first_name': ['John', 'Jane', 'Joe'],
    'last_name': ['Doe', 'Doe', 'Bloggs'],
    '2018': [20000, 20000, 20000],
    '2019': [21000, 21000, 21000],
    '2020': [22000, 22000, 22000]
})))
###

# Error in code      SystemError: Negative size passed to PyUnicode_New
