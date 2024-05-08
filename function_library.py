import datetime
import pandas
import numpy

def get_today_date() -> datetime.date:
    """
    Context:
    Get today's date.

    Parameters:
    None -> No parameters required.

    Returns:
    "today_date": datetime.date -> Today's date.
    """
    today_date = datetime.date.today()
    return today_date

def calculate_date_difference(start_date: datetime.date, end_date: datetime.date) -> int:
    """
    Context:
    Calculate the difference in the number of days between two dates.

    Parameters:
    "start_date": datetime.date -> The starting date.
    "end_date": datetime.date -> The ending date.

    Returns:
    "day_difference": int -> The number of days between "start_date" and "end_date".
    """
    day_difference = (end_date - start_date).days
    return day_difference

def load_csv_file(file_name: str) -> pandas.DataFrame:
    """
    Context:
    Load a CSV file and return its data as a DataFrame.

    Parameters:
    "file_name": str -> Name of the CSV file to load.

    Returns:
    "csv_dataframe": pandas.DataFrame -> DataFrame of the loaded CSV file.
    """
    csv_dataframe = pandas.read_csv(file_path)
    return csv_dataframe

def get_dataframe_column(csv_dataframe: pandas.DataFrame, column_id: str) -> pandas.DataFrame:
    """
    Context:
    Get a specified column from a given DataFrame object.

    Parameters:
    "csv_dataframe": pandas.DataFrame -> CSV DataFrame that is being filtered.
    "column_id": str -> Column label within the CSV DataFrame to filter.

    Returns:
    "column_dataframe": pandas.DataFrame -> Filtered column of the CSV DataFrame.
    """

    if column_id in csv_dataframe.columns:
        column_dataframe = csv_datafram.get(column_id)
        return column_dataframe
    else:
        # Handle the case where the specified column_id is not found
        print(f"Error: Column '{column_id}' not found in the DataFrame.")
        return None

def get_dataframe_row(csv_dataframe: pandas.DataFrame, row_id: str) -> pandas.DataFrame:
    """
    Context:
    Get a specified row from a given DataFrame object.

    Parameters:
    "csv_dataframe": pandas.DataFrame -> CSV DataFrame that is being filtered.
    "row_id": str -> Row label within the CSV DataFrame to filter.

    Returns:
    "row_dataframe": pandas.DataFrame -> Filtered row of the CSV DataFrame.
    """
    if row_id in csv_dataframe.index:
        row_dataframe = csv_dataframe.loc[[row_id]]
        return row_dataframe
    else:
        print(f"Error: Row '{row_id}' not found in the DataFrame.")
        return None

def convert_dataframe_to_array(dataframe_to_convert: pandas.DataFrame) -> list:
    """
    Context:
    Convert a given DataFrame object to a NumPy array.

    Parameters:
    "dataframe_to_convert": pandas.DataFrame -> The input dataframe.

    Returns:
    "dataframe_array": numpy.ndarray -> Numpy array representation of the dataframe.
    """
    dataframe_array = dataframe_to_convert.to_numpy()
    return dataframe_array

def calculate_dataframe_sum(dataframe_to_sum: pandas.DataFrame) -> float:
    """
    Context:
    Sum all values within the dataframe.

    Parameters:
    "dataframe_to_sum": pandas.DataFrame -> The input dataframe.

    Returns:
    "dataframe_sum": float -> The sum of all values in the dataframe.
    """
    dataframe_sum = dataframe_to_sum.values.sum()
    return dataframe_sum

def calculate_dataframe_average(dataframe_to_average: pandas.DataFrame) -> float:
    """
    Context:
    Get the average of the values within the dataframe.

    Parameters:
    "dataframe_to_average": pandas.DataFrame -> The input dataframe.

    Returns:
    "dataframe_average": float -> The average of all values in the dataframe.
    """
    dataframe_average = dataframe_to_average.values.mean()
    return dataframe_average

def convert_dataframe_datetime(dataframe_to_convert: pandas.DataFrame) -> pandas.Series:
    """
    Context:
    Convert all values in a given DataFrame object to datetime.

    Parameters:
    "dataframe_to_convert": pandas.DataFrame -> The input dataframe.

    Returns:
    "series_datetime": pandas.Series -> A series with datetime values.
    """
    series_datetime = pandas.to_datetime(dataframe_to_convert, errors='coerce')
    return series_datetime

def get_series_min_index(series_to_min: pandas.Series) -> int:
    """
    Context:
    Get the minimum value's index from a given series object.

    Parameters:
    "series_to_min": pandas.Series -> The input series.

    Returns:
    "min_index": int -> The index of the minimum value in the series.
    """
    min_index = series_to_min.idxmin()
    return min_index

def get_series_max_index(series_to_max: pandas.Series) -> int:
    """
    Context:
    Get the maximum value's index from a given series object.

    Parameters:
    "series_to_max": pandas.Series -> The input series.

    Returns:
    "max_index": int -> The index of the maximum value in the series.
    """
    max_index = series_to_max.idxmax()
    return max_index


# Demo testing


def closest_date_to_today(file_name: str) -> pandas.DataFrame:
    """
    Context:
    Open a CSV file, convert all rows to datetime values, and find the row closest to today's date.

    Parameters:
    "file_name": str -> Name of the CSV file to open.

    Returns:
    "closest_row": pandas.DataFrame -> The row in the CSV file closest to today's date.
    """

    # Step 1: Load CSV file into a DataFrame
    csv_dataframe = load_csv_file(file_name)

    # Step 2: Convert all values in the DataFrame to datetime
    datetime_series = convert_dataframe_datetime(csv_dataframe)

    # Step 3: Calculate the difference in days between each row's date and today's date
    today = get_today_date()
    date_differences = datetime_series.apply(lambda row: calculate_date_difference(row.date(), today))

    # Step 4: Find the index of the row with the minimum date difference (closest to today)
    min_index = get_series_min_index(date_differences)

    # Step 5: Retrieve the row with the minimum date difference
    closest_row = get_dataframe_row(csv_dataframe, min_index)

    return closest_row



































