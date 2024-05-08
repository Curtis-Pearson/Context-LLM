import pandas as pd

def read_csv(file_name: str) -> pd.DataFrame:
    """
    Context:
    This function is used to read data from the CSV file.

    Parameters:
    "file_name": str -> The name of the CSV file to read data from.

    Returns:
    "data": pd.DataFrame -> The data read from the CSV file.
    """
    try:
        data: pd.DataFrame = pd.read_csv(file_name)
        return data
    except FileNotFoundError:
        print("The file does not exist.")
        return None

### Test cases
# The CSV file exists and is readable.
print(read_csv('salaries.csv'))

# The CSV file does not exist.
print(read_csv('non_existent.csv'))
###

def calculate_average_salary(data: pd.DataFrame) -> float:
    """
    Context:
    This function is used to calculate the average salary of all employees for the years 2018-2020.

    Parameters:
    "data": pd.DataFrame -> The data read from the CSV file.

    Returns:
    "average_salary": float -> The average salary of all employees for the years 2018-2020.
    """
    try:
        # Filter the data for the years 2018-2020
        filtered_data = data[(data['Year'] >= 2018) & (data['Year'] <= 2020)]
        
        # Calculate the average salary of all employees for these years
        average_salary = filtered_data['Salary'].mean()
        
        return average_salary
    except Exception as e:
        print("Error in calculating average salary: ", str(e))
        return None

### Test cases
# The data contains salary information for the years 2018-2020.
data = pd.DataFrame({
    'Year': [2018, 2019, 2020, 2021],
    'Salary': [50000, 55000, 60000, 65000]
})
print(calculate_average_salary(data))

# The data does not contain salary information for the years 2018-2020.
data = pd.DataFrame({
    'Year': [2015, 2016, 2017],
    'Salary': [40000, 45000, 50000]
})
print(calculate_average_salary(data))
###

def main(file_name: str) -> float:
    """
    Context:
    The purpose of the program is to calculate the average salary of all employees between the years 2018-2020 from a CSV file.

    Parameters:
    "file_name": str -> The name of the CSV file to read data from.

    Returns:
    "average_salary": float -> The average salary of all employees for the years 2018-2020.
    """
    data: pd.DataFrame = read_csv(file_name)
    
    if data is None:
        return None
        
    average_salary: float = calculate_average_salary(data)
    
    return average_salary

### Test cases
# The CSV file exists and is readable, and contains salary information for the years 2018-2020.
print(main('salaries.csv'))

# The CSV file does not exist.
print(main('non_existent.csv'))

# The CSV file exists and is readable, but does not contain salary information for the years 2018-2020.
print(main('salaries_without_2018_2020.csv'))
###
