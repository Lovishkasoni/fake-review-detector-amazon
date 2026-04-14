import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    """TF-IDF vectorizer with phrase mapping"""
    
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            lowercase=True
        )
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, reviews):
        """Fit TF-IDF vectorizer"""
        self.vectorizer.fit(reviews)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.is_fitted = True
        print(f"Fitted vectorizer with {len(self.feature_names)} features")
        return self
    
    def transform(self, reviews):
        """Transform reviews to TF-IDF vectors"""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        return self.vectorizer.transform(reviews).toarray()
    
    def fit_transform(self, reviews):
        """Fit and transform in one step"""
        self.fit(reviews)
        return self.transform(reviews)
    
    def get_feature_names(self):
        """Get feature (phrase) names"""
        return self.feature_names
    
    def get_top_features(self, indices, top_k=5):
        """Get top features given indices"""
        top_indices = np.argsort(indices)[-top_k:][::-1]
        return [(self.feature_names[i], indices[i]) for i in top_indices]

# Usage
if __name__ == "__main__":
    reviews = ["good product", "best product ever", "works well"]
    extractor = FeatureExtractor(max_features=1000)
    features = extractor.fit_transform(reviews)
    print(f"Features shape: {features.shape}")