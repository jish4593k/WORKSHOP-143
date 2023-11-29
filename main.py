import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics
import tkinter as tk
from tkinter import filedialog


digits = load_digits()


scaled_data = scale(digits.data)
target_labels = digits.target


num_clusters = 10

samples, features = scaled_data.shape

def evaluate_clustering(estimator, name, data, labels):
    estimator.fit(data)
    inertia = estimator.inertia_
    homogeneity = metrics.homogeneity_score(labels, estimator.labels_)
    completeness = metrics.completeness_score(labels, estimator.labels_)
    v_measure = metrics.v_measure_score(labels, estimator.labels_)
    adj_rand_score = metrics.adjusted_rand_score(labels, estimator.labels_)
    adj_mutual_info = metrics.adjusted_mutual_info_score(labels, estimator.labels_)
    silhouette_score = metrics.silhouette_score(data, estimator.labels_, metric='euclidean')

    print(f'{name}\tInertia: {inertia:.3f}\tHomogeneity: {homogeneity:.3f}\tCompleteness: {completeness:.3f}\tV-Measure: {v_measure:.3f}\tAdjusted Rand Score: {adj_rand_score:.3f}\tAdjusted Mutual Info: {adj_mutual_info:.3f}\tSilhouette Score: {silhouette_score:.3f}')

# Perform clustering
# n_clusters = number of clusters
# init = method of initialization (Can preset to cut off some training time)
# n_init = number of initializations to perform
kmeans_clf = KMeans(n_clusters=num_clusters, init="random", n_init=100)
evaluate_clustering(kmeans_clf, "KMeans", scaled_data, target_labels)


# Create a tkinter window
window = tk.Tk()
window.title("Digit Clustering with K-Means")
window.geometry("800x600")

def choose_file():
    file_path = filedialog.askopenfilename()
    print(f'Selected file: {file_path}')

file_button = tk.Button(window, text="Choose File", command=choose_file)
file_button.pack()

window.mainloop()
