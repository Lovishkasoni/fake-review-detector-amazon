import numpy as np

class GradientExplainer:
    """Gradient-based explainability for phrase attribution"""
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
    
    def compute_input_gradients(self, X):
        """Compute gradients w.r.t. input"""
        # Forward pass
        output = self.model.forward(X)
        
        # Backward pass to get gradients
        y_true = output.round()  # Use predictions as targets
        self.model.backward(y_true)
        
        # Gradient w.r.t. input
        dA1 = np.dot(self.model.gradients['dZ1'], self.model.W1.T)
        return dA1
    
    def explain_prediction(self, X, top_k=5):
        """Explain a single prediction"""
        # Get gradients
        gradients = self.compute_input_gradients(X)
        
        # Use absolute gradient values as importance
        importance = np.abs(gradients[0])
        
        # Get top features
        top_indices = np.argsort(importance)[-top_k:][::-1]
        
        explanations = []
        for idx in top_indices:
            if idx < len(self.feature_names):
                phrase = self.feature_names[idx]
                impact = importance[idx]
                explanations.append((phrase, impact))
        
        return explanations

# Usage
if __name__ == "__main__":
    print("Explainer module ready")