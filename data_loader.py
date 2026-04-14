import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import re

class AmazonReviewLoader:
    """Load and preprocess Amazon reviews dataset"""
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.train_reviews = None
        self.test_reviews = None
        self.train_labels = None
        self.test_labels = None
    
    def preprocess_text(self, text):
        """Clean and preprocess review text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def create_sample_dataset(self, num_samples=1000):
        """Create a sample dataset with fake and real reviews"""
        np.random.seed(self.random_state)
        
        # Real review phrases
        real_phrases = [
            "good product", "works as expected", "happy with purchase",
            "great quality", "worth the money", "excellent service",
            "recommended", "reliable", "satisfied customer",
            "good value", "solid product", "does what it says"
        ]
        
        # Fake review phrases (suspicious patterns)
        fake_phrases = [
            "best product ever", "must buy now", "highly recommend",
            "amazing", "perfect", "5 stars", "best purchase",
            "incredible deal", "life changing", "absolutely love it",
            "everyone should buy", "buy immediately"
        ]
        
        reviews = []
        labels = []
        
        # Generate real reviews (label = 0)
        for _ in range(num_samples // 2):
            num_phrases = np.random.randint(2, 5)
            review = " ".join(np.random.choice(real_phrases, num_phrases))
            reviews.append(review)
            labels.append(0)
        
        # Generate fake reviews (label = 1)
        for _ in range(num_samples // 2):
            num_phrases = np.random.randint(2, 5)
            review = " ".join(np.random.choice(fake_phrases, num_phrases))
            # Add exclamation marks and emojis text
            review += " " + "!" * np.random.randint(1, 4)
            reviews.append(review)
            labels.append(1)
        
        return reviews, labels
    
    def load_data(self):
        """Load and split data"""
        print("Creating sample dataset...")
        reviews, labels = self.create_sample_dataset(1000)
        
        # Preprocess reviews
        reviews = [self.preprocess_text(r) for r in reviews]
        
        # Split data
        train_reviews, test_reviews, train_labels, test_labels = train_test_split(
            reviews, labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels
        )
        
        self.train_reviews = train_reviews
        self.test_reviews = test_reviews
        self.train_labels = np.array(train_labels)
        self.test_labels = np.array(test_labels)
        
        print(f"Loaded {len(reviews)} reviews")
        print(f"Train set: {len(train_reviews)} samples")
        print(f"Test set: {len(test_reviews)} samples")
        print(f"Fake reviews: {np.sum(self.train_labels)} in train set")
        
        return self.train_reviews, self.test_reviews, self.train_labels, self.test_labels

# Usage
if __name__ == "__main__":
    loader = AmazonReviewLoader()
    train_reviews, test_reviews, train_labels, test_labels = loader.load_data()