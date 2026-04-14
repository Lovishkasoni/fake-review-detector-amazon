import numpy as np
from feature_extractor import FeatureExtractor
from ann_model import ANNModel
from explainer import GradientExplainer
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
        return model
    except:
        print("Model weights not found. Run train.py first!")
        return None

def demo():
    """Interactive inference demo"""
    print("="*70)
    print("FAKE REVIEW DETECTOR - INTERACTIVE DEMO")
    print("="*70)
    
    # Load model and feature extractor
    print("\n[Loading model...]")
    model = load_trained_model()
    if model is None:
        return
    
    # Create feature extractor (fit on training data)
    loader = AmazonReviewLoader()
    train_reviews, _, _, _ = loader.load_data()
    extractor = FeatureExtractor(max_features=5000)
    extractor.fit(train_reviews)
    
    feature_names = extractor.get_feature_names()
    explainer = GradientExplainer(model, feature_names)
    
    print("Model loaded successfully!\n")
    
    # Interactive loop
    while True:
        print("-"*70)
        review = input("\nPaste a review (or 'quit' to exit):\n> ").strip()
        
        if review.lower() == 'quit':
            print("\nGoodbye!")
            break
        
        if not review:
            print("Empty review. Please try again.")
            continue
        
        # Preprocess and extract features
        preprocessed = loader.preprocess_text(review)
        X = extractor.transform([preprocessed])
        
        # Make prediction
        prediction = model.predict(X)[0][0]
        confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
        
        # Get explanations
        explanations = explainer.explain_prediction(X, top_k=5)
        
        # Display results
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        print(f"Original Review: {review[:100]}...")
        print(f"\nFake Probability: {prediction*100:.1f}%")
        print(f"Confidence Score: {confidence:.1f}%")
        print(f"Classification: {'LIKELY FAKE' if prediction > 0.5 else '✓ LIKELY REAL'}")
        
        print(f"\nTop Suspicious Phrases:")
        for i, (phrase, impact) in enumerate(explanations, 1):
            print(f"  {i}. '{phrase}' → impact: +{impact:.3f}")
        print("="*70)

if __name__ == "__main__":
    demo()