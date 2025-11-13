from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np

class KMeansClustering:
    def __init__(self, n_clusters=2, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)

    def fit(self, X):
        """Fit KMeans and return predicted cluster assignments."""
        self.labels_ = self.model.fit_predict(X)
        return self.labels_

    def inertia(self):
        """Return the final inertia (sum of squared distances)."""
        return self.model.inertia_

    def centroids(self):
        return self.model.cluster_centers_

    def score(self, X):
        """Return negative inertia (higher is better)."""
        return -self.model.score(X)

    def evaluation_metrics(self, X, y):
        pred = self.labels_
        ari = adjusted_rand_score(y, pred)
        nmi = normalized_mutual_info_score(y, pred)
        inertia = self.inertia()
        #score = self.score(X)
        return {"ARI": ari, "NMI": nmi, "Inertia": inertia}
        
