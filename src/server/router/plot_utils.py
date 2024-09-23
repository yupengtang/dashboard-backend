import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.express as px
import geopandas as gpd
from shapely.geometry import Point

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull


colors_extended = ['#488f31', '#8ea644', '#cabd65', '#ffd58f', '#f5a66a', '#e87556', '#d43d51',]


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def draw_custers_hull(data, df_fit_pca):
    df_fit_pca[0] = normalize_data(df_fit_pca[0])
    df_fit_pca[1] = normalize_data(df_fit_pca[1])


    fig, ax = plt.subplots(1, figsize=(8,6))
    plt.scatter(
        x = df_fit_pca[0],
        y = df_fit_pca[1],
        s = 20,
        c = df_fit_pca.cl_label)

    plt.scatter(df_fit_pca[0].iloc[0].item(), df_fit_pca[1].iloc[0].item(), c='red', marker='.', s=200)
    plt.annotate('ROGERS PARK', (df_fit_pca[0].iloc[0].item(), df_fit_pca[1].iloc[0].item()))

    # draw enclosure
    clusters = df_fit_pca.cl_label.unique()
    clusters.sort()
    for i in clusters:
        points = df_fit_pca[df_fit_pca.cl_label == i][[0, 1]].values
        # get convex hull
        hull = ConvexHull(points)
        # get x and y coordinates
        # repeat last point to close the polygon
        x_hull = np.append(points[hull.vertices,0],
                        points[hull.vertices,0][0])
        y_hull = np.append(points[hull.vertices,1],
                        points[hull.vertices,1][0])
        # plot shape
        plt.fill(x_hull, y_hull, alpha=0.3, c=colors_extended[i], label=str(i))

    plt.xlabel('PCA component 1')
    plt.ylabel('PCA component 2')
    plt.title('Clustered areas after dimensionality reduction with 2 components')
    plt.legend(ncol=len(clusters))
    plt.margins(x=0.015, y=0.01)


def read_data(filename:str):
    df_raw_data = pd.read_csv(filename)
    return df_raw_data


def preprocess_life_quality_data(df):
    df_life_quality = df.iloc[4:,1:]

    # metadata = df.iloc[0:3, 5:]
    # metadata_df = metadata.transpose()
    # metadata_df = metadata_df.rename(columns={0:"indicator",1:"description",2:"data source"})
    
    df_life_quality.GEOID = pd.to_numeric(df_life_quality.GEOID)
    ind_cols = df_life_quality.columns[4:]
    df_life_quality[ind_cols] = df_life_quality[ind_cols].apply(pd.to_numeric)
    df_life_quality = df_life_quality.reset_index(drop = True)
    
    df_life_quality.Name = df_life_quality.Name.str.upper() 
    df_life_quality.loc[73,'Name'] = 'OHARE'

    return df_life_quality, ind_cols


def perform_pca(X, n_components:int = 2):
    pca = PCA(n_components=n_components)
    fit_pca = pca.fit_transform(X)
    return fit_pca


def kmeans_life_quality(df_life_quality, ind_cols, k:int = 5):
    '''
    This method takes pd.DataFrame with life quality metrics and columns to select for clustering,
    performs K-Means and PCA for plotting.

    :param df_life_quality: pandas DataFrame with life quality metrics
    :param ind_cols: columns to select for clustering
    :param k : number of clusters
    :return df_life_quality : pandas DataFrame with life quality metrics and clusters assigned
    :return df_fit_pca : PCA result (nx+2 dim)
    '''
    # impute
    X = df_life_quality[ind_cols]
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X)
    X_imp = imp.transform(X)

    # reduce to 2d for plot
    fit_pca = perform_pca(X_imp, 2)
    df_fit_pca = pd.DataFrame(fit_pca)

    kmeans = KMeans(n_clusters=k, random_state=0).fit(X_imp)
    df_life_quality['cl_label'] = kmeans.labels_
    df_fit_pca['cl_label'] = kmeans.labels_

    return df_life_quality, df_fit_pca 


def plot_clusters(df_fit_pca, park_to_mark_id:int, park_name:str):
    df_fit_pca[0] = normalize_data(df_fit_pca[0])
    df_fit_pca[1] = normalize_data(df_fit_pca[1])

    # plot data
    fig, ax = plt.subplots(1, figsize=(8,6))
    
    plt.scatter(x = df_fit_pca[0], y = df_fit_pca[1], s = 20, c = df_fit_pca.cl_label)

    # mark one point with red
    if park_to_mark_id and park_name != '':
        plt.scatter(df_fit_pca[0].iloc[park_to_mark_id].item(), df_fit_pca[1].iloc[park_to_mark_id].item(), c='red', marker='.', s=200)
        plt.annotate(park_name, (df_fit_pca[0].iloc[park_to_mark_id].item(), df_fit_pca[1].iloc[park_to_mark_id].item()))

    # draw enclosure
    clusters = df_fit_pca.cl_label.unique()
    clusters.sort()
    for i in clusters:
        points = df_fit_pca[df_fit_pca.cl_label == i][[0, 1]].values
        hull = ConvexHull(points)
        x_hull = np.append(points[hull.vertices,0], points[hull.vertices,0][0])
        y_hull = np.append(points[hull.vertices,1], points[hull.vertices,1][0])
        plt.fill(x_hull, y_hull, alpha=0.3, c=colors_extended[i], label=str(i))

    plt.xlabel('PCA component 1')
    plt.ylabel('PCA component 2')
    plt.title('Clustered areas after dimensionality reduction with 2 components')
    plt.legend(ncol=len(clusters))
    plt.margins(x=0.015, y=0.01)
    plt.show()
     