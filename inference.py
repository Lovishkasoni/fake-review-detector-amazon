# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from feature_extractor import FeatureExtractor
from ann_model import ANNModel
from data_loader import AmazonReviewLoader

def load_trained_model():
    """Load trained model weights"""
    try:
        weights = np.load('model_weights.npy', allow_pickle=True).item()
        model = ANNModel(input_dim=5000)
        model.W1 = weights['W1']
        model.b1 = weights['b1']
        model.W2 = weights['W2']
        model.b2 = weights['b2']
        model.W3 = weights['W3']
        model.b3 = weights['b3']
        print("[OK] Trained model loaded!")
        return model
    except FileNotFoundError:
        print("[INFO] Model weights not found. Using untrained model.")
        return ANNModel(input_dim=5000)

def demo():
    """Interactive inference demo"""
    print("="*70)
    print("FAKE REVIEW DETECTOR - INTERACTIVE DEMO")
    print("="*70)
    
    # Load model and feature extractor
    print("\n[Loading model...]")
    model = load_trained_model()
    
    # Create feature extractor
    print("[Creating feature extractor...]")
    loader = AmazonReviewLoader()
    train_reviews, _, _, _ = loader.load_data()
    
    extractor = FeatureExtractor(max_features=5000)
    extractor.fit(train_reviews)
    feature_names = extractor.get_feature_names()
    print("[OK] System ready!\n")
    
    # Interactive loop
    while True:
        print("-"*70)
        review = input("\nPaste a review (or 'quit' to exit):\n> ").strip()
        
        if review.lower() == 'quit':
            print("\nGoodbye!")
            break
        
        if not review:
            print("[ERROR] Empty review. Please try again.")
            continue
        
        try:
            print("\n[Processing...]")
            
            # Preprocess and extract features
            preprocessed = loader.preprocess_text(review)
            X = extractor.transform([preprocessed])
            
            print(f"[OK] Features extracted: {X.shape}")
            
            # Make prediction
            print("[Making prediction...]")
            prediction = model.predict(X)[0][0]
            
            # Calculate confidence
            if prediction > 0.5:
                confidence = prediction * 100
                classification = "LIKELY FAKE"
                symbol = "[FAKE]"
            else:
                confidence = (1 - prediction) * 100
                classification = "LIKELY REAL"
                symbol = "[REAL]"
            
            # Display results
            print("\n" + "="*70)
            print("PREDICTION RESULTS")
            print("="*70)
            print(f"Review: {review[:80]}...")
            print(f"\nFake Probability: {prediction*100:.1f}%")
            print(f"Confidence Score: {confidence:.1f}%")
            print(f"Classification: {symbol} {classification}")
            
            # Show top phrases
            importance = X[0]
            top_indices = np.argsort(importance)[-5:][::-1]
            
            print(f"\nTop Phrases Found:")
            for i, idx in enumerate(top_indices, 1):
                if idx < len(feature_names) and importance[idx] > 0:
                    phrase = feature_names[idx]
                    score = importance[idx]
                    print(f"  {i}. '{phrase}' → score: {score:.3f}")
            
            print("="*70)
            
        except Exception as e:
            print(f"\n[ERROR] Prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    demo()