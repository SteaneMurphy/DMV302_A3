import csv
import numpy as np
import pandas as pd



###########################
#       CSV UTILITY       #
###########################

def read_csv(path:str, skip_header:bool = False):
    """
    Reads a CSV file and returns non-empty rows.

    Args:
        path (str): filepath to CSV
        skip_header (bool): whether to skip first row

    Returns:
        list[str]: a single CSV row as a list of strings
    """
    with open(path, 'r') as file:                                                   # open csv file
        reader = csv.reader(file)                                                   # read file

        if skip_header:                                                             # optional skip first row
            next(reader)

        for row in reader:                                                          # for each row return row if not empty
            if row:
                yield row



def convert_to_date(dates: list[str]) -> pd.DatetimeIndex:
    """
    Converts a list of date strings into pandas datetime format.

    The input dates are expected to be in the format:
        day/month/year (dd/mm/yy)

    Args:
        dates (list[str]): List of date strings from the dataset

    Returns:
        pd.DatetimeIndex: Pandas datetime
    """
    return pd.to_datetime(dates, format='%d/%m/%y')                                 # convert to pandas datetime object



def parse_numeric(rows:list[str]) -> np.ndarray[float]:
    """
    Parses rows of purely numeric values.

    Args:
        rows: array of CSV rows

    Returns:
        np.ndarray (float): array of floats
    """
    return np.array([[float(x.strip()) for x in row] for row in rows])              # strip and convert to float for each row in given dataset



###########################
#     TASK 1 UTILITY      #
###########################

def parse_rows_x_y(rows:list[str]) -> np.ndarray[float]:
    """
    Parse rows of purely numeric values and
    returns the x and y columns only.

    Args:
        rows: array of CSV rows

    Returns:
        np.ndarray: shape (N, 2) containing x and y columns
    """
    return np.array([                                                               # strip and convert to float for each row in given dataset
        [float(row[0].strip()), float(row[1].strip())]
        for row in rows
    ])   



###########################
#     TASK 2 UTILITY      #
###########################

def split_date_column(rows: list[list[str]]) -> tuple:
    """
    Splits a dataset into a separate date column and numeric feature columns.

    Args:
        rows (list[list[str]]): Raw CSV rows where each row contains:
            - index 0: date string
            - index 1..n: numeric feature values (as strings)

    Returns:
        tuple:
            - list[str]: List of date strings
            - list[list[str]]: Remaining numeric data (still as strings)
    """
    dates = [row[0] for row in rows]                                               # store all values from row 0 (dates)
    numeric_rows = [row[1:] for row in rows]                                       # store all values from proceeding rows (features)
    return dates, numeric_rows



def split_rows(features: np.ndarray) -> dict:
    """
    Splits a numeric feature matrix into a dictionary of named columns
    for easier access during feature engineering and visualisation.

    Args:
        features (np.ndarray): 2D array where each row is a daily record
            and each column represents a specific weather variable

    Returns:
        dict: Dictionary mapping feature names to numpy arrays
    """
    return {
        "temp_avg": features[:, 0],                                                # average temperature
        "temp_min": features[:, 1],                                                # minimum temperature
        "temp_max": features[:, 2],                                                # maximum temparature

        "temp_min_hist": features[:, 3],                                           # average minimum temperature (historical)
        "temp_max_hist": features[:, 4],                                           # average maximum temparature (historical)
        "temp_low_hist": features[:, 5],                                           # actual minimum temperature (historical)
        "temp_high_hist": features[:, 6],                                          # actual maximum temperature (historical)

        "rain": features[:, 7],                                                    # rainfall amount
        "rain_avg_hist": features[:, 8],                                           # average rainfall (historical)
        "rain_max_hist": features[:, 9],                                           # highest rainfaill (historical)
    }
 


###########################
#  PREPROCESSING UTILITY  #
###########################

def calculate_mean(features: np.ndarray) -> np.ndarray:
    return np.mean(features, axis=0)

def calculate_standard_deviation(features: np.ndarray) -> np.ndarray:
    return np.std(features, axis=0)

def standardise(features: np.ndarray) -> np.ndarray:
    """
    Standardises the feature data so each column has mean 0 and standard deviation 1.

    Args:
        features (np.ndarray): 2D array where each row is a sample and each column is a feature

    Returns:
        np.ndarray: standardised features
        mean
        std
    """
    mean = calculate_mean(features)                                                 # calculate mean
    std = calculate_standard_deviation(features)                                    # calculate standard deviation
    std = np.where(std == 0, 1, std)                                                # edge-case, division by 0

    features = (features - mean) / std                                              # standardise data (z-score scaling)    
    return features, mean, std                                          