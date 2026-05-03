import matplotlib.pyplot as pyplot
from matplotlib import gridspec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from utility.util import read_csv, parse_rows_x_y, standardise



"""
Preprocess and standardise given numeric data set.

Args:
    path (str): filepath to the dataset

Returns:
    features (np.ndarray): array of standardised data
"""
def preprocess_data(path:str) -> np.ndarray:
    rows = read_csv(path, skip_header=True)                                                    # import rows from CSV
    features = parse_rows_x_y(rows)                                                            # parse rows into numeric data format
    features, mean, std = standardise(features)                                                # standardise the rows for clustering

    return features



"""
Creates a 2D projection plot on 2 given axes (x, y).

- Preprocess data
- Splits data into axes
- Creates graph/plot layout
- Assigns data to scatterplot
- Assigns data to histograms
"""
def create_x_y_projection_plot():
    features = preprocess_data("data/DMVA3T1.csv")                                             # preprocess data into features set

    x = features[:, 0]                                                                         # split out x-axis column
    y = features[:, 1]                                                                         # split out y-axis column

    figure = pyplot.figure(figsize=(8, 8))                                                     # create new pyplot

    axis_hist_x, axis_hist_y, scatterplot = create_layout(figure)                              # create layout (two histograms, one scatterplot, labels)
    plot_scatter(x, y, scatterplot)                                                            # create scatterplot
    plot_histograms(axis_hist_x, axis_hist_y, x, y)                                            # create historgrams for both axes

    pyplot.savefig("DMV302T1a.png")                                                            # save pyplot as image
    pyplot.close()                                                                          



"""
Creates the grid container for the scatterplot
and histograms (layout).

- top (histogram)
- middle left (scatterplot)
- middle right (histogram)
"""
def create_layout(figure: Figure) -> tuple:
    grid = gridspec.GridSpec(                                                                  # create grid container
        2, 2,
        width_ratios=[4, 1],
        height_ratios=[1, 4],
        hspace=0.25,
        wspace=0.15
    )

    axis_hist_x = figure.add_subplot(grid[0, 0])                                               # add x-axis histogram container (top)
    axis_hist_y = figure.add_subplot(grid[1, 1])                                               # add y-axis histogram container (middle-right)
    scatterplot = figure.add_subplot(grid[1, 0])                                               # add scatterplot position container (middle-left)

    return axis_hist_x, axis_hist_y, scatterplot



"""
Assigns the standardised data to a scatterplot graph
based on each data points (x, y) co-ords.
"""
def plot_scatter(x: np.ndarray, y: np.ndarray, scatterplot: Axes):
    scatterplot.scatter(x, y, color="black", alpha=0.5, s=30)                                  # create scatterplot from given data
    scatterplot.set_title("2D Projection (X-Y)", pad=12)                                       # scatterplot heading
    scatterplot.set_xlabel("X", labelpad=10)                                                   # x-axis label
    scatterplot.set_ylabel("Y", labelpad=10)                                                   # y-axis label



"""
Assigns each data point to its associated axes and plots on
a histogram based on frequency in the data set.
"""
def plot_histograms(axis_hist_x: Axes, axis_hist_y: Axes, x: np.ndarray, y: np.ndarray):
    axis_hist_x.hist(x, bins=30, color="#6baed6", edgecolor="white", linewidth=0.5)          # create x-axis histogram
    axis_hist_x.set_ylabel("Frequency", labelpad=10)                                           # y-axis label
    axis_hist_x.tick_params(axis="x", labelbottom=False)                                       # histogram bar spacing

    axis_hist_y.hist(y, bins=30, orientation="horizontal", color="#fc9272",                  # create y-axis histogram
                     edgecolor="white", linewidth=0.5)
    axis_hist_y.set_xlabel("Frequency", labelpad=10)                                           # x-axis label
    axis_hist_y.tick_params(axis="y", labelleft=False)                                         # histogram bar spacing



create_x_y_projection_plot()