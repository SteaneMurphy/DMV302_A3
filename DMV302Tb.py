import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.dates as mdates
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utility.util import (
    read_csv,
    parse_numeric,
    split_date_column,
    convert_to_date,
    split_rows
)



"""
Preprocess and standardise given numeric data set.

Args:
    path (str): filepath to the dataset

Returns:
    dates
    features (np.ndarray): array of standardised data
"""
def preprocess_data(path: str):
    rows = list(read_csv(path, skip_header=True))
    dates, rows = split_date_column(rows)
    features = parse_numeric(rows)
    return dates, features



def run_weather_analysis_pipeline():
    dates, numeric_data = preprocess_data("data/DMVA3T2.csv")                   # preprocess data from csv, splits date column from features
    dates = convert_to_date(dates)                                              # convert date column to readable datetime format type
    split = split_rows(numeric_data)                                            # create named rows from feature data for analysis
    features = identify_features(split)                                         # create engineered features from base feature data

    cluster_labels = cluster_weather(features)                                  # cluster weather data
    visualize_weather(dates, split, features)                                   # create visualisations for engineered features
    plot_clusters(features, cluster_labels)                                     # create visualisation for clustered data



###########################
#   FEATURE ENGINEERING   #
###########################

def extract_weather_features(features: dict):
    """
    Extracts raw weather variables from the feature dictionary.

    Args:
        features (dict): Dictionary of weather variables from split_rows()

    Returns:
        tuple: unpacked weather variables for feature engineering
    """

    temp_avg = features["temp_avg"]
    temp_min = features["temp_min"]
    temp_max = features["temp_max"]

    temp_min_hist = features["temp_min_hist"]
    temp_max_hist = features["temp_max_hist"]

    rain = features["rain"]
    rain_avg_hist = features["rain_avg_hist"]
    rain_max_hist = features["rain_max_hist"]

    return (
        temp_avg,
        temp_min,
        temp_max,
        temp_min_hist,
        temp_max_hist,
        rain,
        rain_avg_hist,
        rain_max_hist
    )



def identify_features(features: dict) -> np.ndarray:
    """
    Performs feature engineering on weather variables by deriving
    additional metrics from raw and historical data.

    The function computes:
    - temperature range (max - min)
    - deviation from historical temperature averages
    - rainfall deviation from historical averages
    - rainfall intensity ratio relative to historical maximum
    """
    (
        temp_avg,
        temp_min,
        temp_max,
        temp_min_hist,
        temp_max_hist,
        rain,
        rain_avg_hist,
        rain_max_hist
    ) = extract_weather_features(features)

    temp_range = temp_max - temp_min
    temp_hist_avg = (temp_min_hist + temp_max_hist) / 2
    temp_dev = temp_avg - temp_hist_avg

    rain_dev = rain - rain_avg_hist
    rain_ratio = rain / (rain_max_hist + 1e-6)

    return np.column_stack([
        temp_avg,
        temp_range,
        temp_dev,
        rain,
        rain_dev,
        rain_ratio
    ])



###########################
#       CLUSTERING        #
###########################

def cluster_weather(features: np.ndarray):
    """
    Applies K-Means clustering to weather feature data to identify
    distinct weather regimes based on temperature and rainfall deviation.

    The function selects two engineered features:
    - Temperature deviation from historical baseline
    - Rainfall deviation from historical baseline

    These features are standardised before clustering to ensure
    equal contribution to the distance metric.
    """
    data = np.column_stack([
        features[:, 2],                                         # temp_dev (tempature deviation)
        features[:, 4]                                          # rain_dev (rainfall deviation)
    ])

    data = StandardScaler().fit_transform(data)                 # standardise dat before clustering

    model = KMeans(n_clusters=3, random_state=42, n_init=10)    # cluster (k=3)
    labels = model.fit_predict(data)

    return labels



###########################
#     VISUALISATION       #
###########################

def visualize_weather(dates, split, features):
    """
    Generates all weather-related visualisations for the dataset,
    combining temperature, rainfall, and derived feature analysis.

    This function acts as a wrapper that calls individual plotting
    functions to produce a complete set of weather analysis figures.

    The visualisations include:
    - Temperature vs historical baseline (including anomaly detection)
    - Rainfall vs historical baseline (including extreme rainfall events)
    - Temperature vs rainfall relationship
    """
    plot_temperature_vs_history(dates, split, features)      
    plot_rainfall_vs_history(dates, split, features)
    plot_temperature_vs_rain(split)



def plot_temperature_vs_history(dates, split, features):
    """
    Visualises daily temperature compared against historical temperature patterns,
    highlighting deviations and temperature anomalies.
    """
    fig, axis = pyplot.subplots(figsize=(14, 4))

    axis.fill_between(
        dates,
        split["temp_low_hist"],
        split["temp_high_hist"],
        color='lightgray',
        alpha=0.4,
        label='Historical Range'
    )

    axis.plot(
        dates,
        split["temp_avg"],
        color='red',
        linewidth=1,
        label='Actual Avg Temp'
    )

    hist_mean = (split["temp_low_hist"] + split["temp_high_hist"]) / 2

    axis.plot(
        dates,
        hist_mean,
        color='black',
        linestyle='--',
        linewidth=1,
        label='Historical Mean'
    )

    temp_dev = features[:, 2]
    warm_anomalies = temp_dev > 10

    axis.scatter(
        dates[warm_anomalies],
        split["temp_avg"][warm_anomalies],
        color='darkred',
        marker='^',
        label='Heat Spike'
    )

    axis.set_title("Temperature vs Historical Baseline")
    axis.set_ylabel("Temperature")
    axis.legend()
    axis.grid(True, alpha=0.3)

    pyplot.tight_layout()
    pyplot.savefig("figures/DMV302T2_temp_vs_history.png")
    pyplot.close()



def plot_rainfall_vs_history(dates, split, features):
    """
    Visualises daily rainfall compared against historical rainfall patterns,
    highlighting deviations and extreme rainfall events.

    The plot includes:
    - Daily observed rainfall (bar chart)
    - Historical average rainfall (line plot)
    - Extreme rainfall events based on rainfall ratio threshold
    """
    fig, axis = pyplot.subplots(figsize=(14, 4))

    axis.bar(
        dates,
        split["rain"],
        color='skyblue',
        label='Daily Rainfall'
    )

    axis.plot(
        dates,
        split["rain_avg_hist"],
        color='navy',
        linestyle='--',
        label='Historical Avg Rain'
    )

    rain_ratio = features[:, 5]
    extreme_rain = rain_ratio > 0.5

    axis.scatter(
        dates[extreme_rain],
        split["rain"][extreme_rain],
        color='blue',
        marker='o',
        s=50,
        label='High Rain Event'
    )

    axis.set_title("Rainfall vs Historical Baseline")
    axis.set_ylabel("Rainfall (mm)")
    axis.legend()
    axis.grid(True, alpha=0.3)

    axis.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    pyplot.xticks(rotation=45)

    pyplot.tight_layout()
    pyplot.savefig("figures/DMV302T2_rain_vs_history.png")
    pyplot.close()



def plot_temperature_vs_rain(split):
    """
    Visualises the relationship between temperature and rainfall
    to explore potential correlation patterns.

    This scatter plot compares:
    - Average daily temperature
    - Daily rainfall
    """
    fig, axis = pyplot.subplots(figsize=(6, 6))

    axis.scatter(
        split["temp_avg"],
        split["rain"],
        alpha=0.5
    )

    axis.set_title("Temperature vs Rainfall Relationship")
    axis.set_xlabel("Temperature")
    axis.set_ylabel("Rainfall (mm)")
    axis.grid(True, alpha=0.3)

    pyplot.tight_layout()
    pyplot.savefig("figures/DMV302T2_temp_vs_rain.png")
    pyplot.close()




def plot_clusters(features, cluster_labels):
    """
    Visualises K-Means clustering results based on weather anomalies,
    specifically temperature and rainfall deviations.

    The plot shows:
    - Temperature deviation vs rainfall deviation
    - Cluster assignments produced by KMeans
    - Colour-coded grouping of weather regimes
    """
    temp_dev = features[:, 2]
    rain_dev = features[:, 4]

    pyplot.figure(figsize=(8, 6))

    scatter = pyplot.scatter(
        temp_dev,
        rain_dev,
        c=cluster_labels,
        cmap='viridis',
        alpha=0.7
    )

    pyplot.xlabel("Temperature Deviation")
    pyplot.ylabel("Rainfall Deviation")
    pyplot.title("Weather Regimes (KMeans Clustering)")
    pyplot.grid(True, alpha=0.3)

    pyplot.colorbar(scatter, label="Cluster")

    pyplot.savefig("figures/DMV302T2b_clusters.png")
    pyplot.close()



run_weather_analysis_pipeline()