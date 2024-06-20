import umap
from sklearn.base import BaseEstimator, TransformerMixin

# Custom estimator for UMAP to be used in GridSearchCV
class UMAPEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.metric = metric
        self.umap = umap.UMAP(n_neighbors=self.n_neighbors, min_dist=self.min_dist, 
                              n_components=self.n_components, metric=self.metric, random_state=42)

    def fit(self, X, y=None):
        self.umap.fit(X, y)
        return self

    def transform(self, X):
        return self.umap.transform(X)
    
    def fit_transform(self, X, y=None):
        return self.umap.fit_transform(X)
