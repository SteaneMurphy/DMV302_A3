import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from utility.util import read_csv, parse_numeric, standardise



def preprocess_data(path:str) -> np.ndarray:
    rows = read_csv(path, skip_header=True)                                                    # import rows from CSV
    features = parse_numeric(rows)                                                             # parse rows into numeric data format
    features, mean, std = standardise(features)                                                # standardise the rows for clustering

    return features



def create_cluster_plot():
    data = preprocess_data("data/DMVA3T1.csv")
    data, labels, centroids = cluster_data(data)
    create_3D_projection_plot(data, labels, centroids)



def create_3D_projection_plot(data: np.ndarray, labels: np.ndarray, centroids):
    fig = pyplot.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    cmap = ListedColormap(pyplot.colormaps["Set2"].colors[:3])

    ax.scatter(
        data[:, 0],
        data[:, 1],
        data[:, 2],
        c=labels,
        cmap=cmap,
        s=20,
        alpha=0.8
    )

    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        centroids[:, 2],
        c="black",
        s=100,
        marker="D",
        label="Centroids"
    )

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}',
            markerfacecolor=cmap(i), markersize=8)
        for i in range(3)
    ]

    legend_elements.append(
        Line2D([0], [0], marker='D', color='w', label='Centroids',
            markerfacecolor='black', markersize=10)
    )

    ax.legend(handles=legend_elements, title="Clusters")

    ax.set_title("3D KMeans Clustering (k=3)", pad=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(elev=30, azim=45)

    pyplot.savefig("DMV302T1b.png")
    pyplot.close()



def cluster_data(data: np.ndarray) -> tuple:
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = model.fit_predict(data)
    centroids = model.cluster_centers_
    return data, labels, centroids



create_cluster_plot()