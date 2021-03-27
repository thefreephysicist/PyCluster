from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS
import hdbscan

class Clusterer:

    def __init__(self):
        self.data = []
        self.min_points = 3
        self.n_clusters = -1;
        self.n_noise = -1;

    def _normalize_data(self):
        scaler = StandardScaler()
        self.normalized_data = scaler.fit(self.data).transform(self.data);

    def set_data(self, input_data):
        self.data = input_data
        self._normalize_data()

    def cluster_hdbscan(self):
        hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_points, gen_min_span_tree=True)
        self.labels = hdbscan_clusterer.fit_predict(self.normalized_data )

        self.n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        self.n_noise = list(self.labels).count(-1)

    """
    Get cluster by one based id
    """
    def get_cluster(self, id):
        indices = [k for k, x in enumerate(self.labels) if x == id]
        return indices



