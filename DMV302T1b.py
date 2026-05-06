import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from utility.util import read_csv, parse_numeric, standardise


"""
Preprocess and standardise given numeric data set.

Args:
    path (str): filepath to the dataset

Returns:
    features (np.ndarray): array of standardised data
"""
def preprocess_data(path:str) -> np.ndarray:
    rows = read_csv(path, skip_header=True)                                                    # import rows from CSV
    features = parse_numeric(rows)                                                             # parse rows into numeric data format
    features, mean, std = standardise(features)                                                # standardise the rows for clustering

    return features



"""
Main clustering plot loop.

- Preprocess data
- Cluster standardised data
- Create 3D projection scatterplot
"""
def create_cluster_plot():
    data = preprocess_data("data/DMVA3T1.csv")                                                
    data, labels, centroids = cluster_data(data)                                               
    create_3D_projection_plot(data, labels, centroids)                                         



"""
Creates a 3D projection scatterplot using the Matplotlib library.

Each data row (x, y, z) is mapped to a new colour, as is each
cluster's centroid.
"""
def create_3D_projection_plot(data: np.ndarray, labels: np.ndarray, centroids):
    figure = pyplot.figure(figsize=(8, 8))                                              # plot dimensions
    scatterplot = figure.add_subplot(111, projection='3d')                              # create scatterplot container
    cmap = ListedColormap(pyplot.colormaps["Set2"].colors[:3])                          # assign colour set

    scatterplot.scatter(                                                                # plot clustered 3D data points
        data[:, 0],
        data[:, 1],
        data[:, 2],
        c=labels,
        cmap=cmap,
        s=20,
        alpha=0.8
    )

    scatterplot.scatter(                                                                # assigns each centroid to cluster
        centroids[:, 0],
        centroids[:, 1],
        centroids[:, 2],
        c="black",
        s=100,
        marker="D",
        label="Centroids"
    )

    legend_elements = [                                                                 # legend setup (data points)
        Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}',
            markerfacecolor=cmap(i), markersize=8)
        for i in range(3)
    ]

    legend_elements.append(                                                             # legend setup (centroids)
        Line2D([0], [0], marker='D', color='w', label='Centroids',
            markerfacecolor='black', markersize=10)
    )

    scatterplot.legend(handles=legend_elements, title="Clusters")                       # headings and labels
    scatterplot.set_title("3D KMeans Clustering (k=3)", pad=10)                         
    scatterplot.set_xlabel("X")
    scatterplot.set_ylabel("Y")
    scatterplot.set_zlabel("Z")

    scatterplot.view_init(elev=30, azim=45)                                             # 3D scatterplot angle and rotation

    pyplot.savefig("figures/DMV302T1b.png")
    pyplot.close()



"""
Clusters given data set into 'n' clusters using KMeans.

Returns clustered data, labels and calculated centroids
"""
def cluster_data(data: np.ndarray) -> tuple:
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = model.fit_predict(data)
    centroids = model.cluster_centers_
    return data, labels, centroids



create_cluster_plot()