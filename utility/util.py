import csv
import numpy as np

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


def parse_numeric(rows:list[str]) -> np.ndarray[float]:
    """
    Parses rows of purely numeric values.

    Args:
        rows: array of CSV rows

    Returns:
        np.ndarray (float): array of floats
    """
    return np.array([[float(x.strip()) for x in row] for row in rows])              # strip and convert to float for each row in given dataset

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

def parse_features_labels(rows:list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Parses rows into features and labels.

    Assumes last column is an integer class label.

    Args:
        rows: array of CSV rows

    Returns:
        np.ndarray (float): features
        np.ndarray (int): labels 
    """
    features = [[float(x.strip()) for x in row[:-1]] for row in rows]               # strip and convert to float for each row in given dataset (minus label column)
    labels = [int(row[-1].strip()) for row in rows]                                 # strip and convert to int for each label column per row
    return np.array(features), np.array(labels)

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

###########################
#      MATH UTILITY       #
###########################

def euclidean_distance(point_a:np.ndarray, point_b:np.ndarray) -> float:
    """
    Calculates the Euclidean distance between two data points.

    This is used to measure similarity between feature vectors,
    where smaller values indicate closer points in feature space.

    Numpy function wrapped to provide readibility.

    Args:
        point_a (np.ndarray): first data point
        point_b (np.ndarray): second data point

    Returns:
        float: straight-line distance between the two points
    """
    return np.linalg.norm(point_a - point_b)                                        # numpy euclidean distance formula