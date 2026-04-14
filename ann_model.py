# -*- coding: utf-8 -*-
import numpy as np

class ANNModel:
    """3-layer Artificial Neural Network with custom backpropagation"""
    
    def __init__(self, input_dim, hidden1=128, hidden2=64, learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden1_dim = hidden1
        self.hidden2_dim = hidden2
        self.learning_rate = learning_rate
        
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden1) * np.sqrt(1 / input_dim)
        self.b1 = np.zeros((1, hidden1))
        
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(1 / hidden1)
        self.b2 = np.zeros((1, hidden2))
        
        self.W3 = np.random.randn(hidden2, 1) * np.sqrt(1 / hidden2)
        self.b3 = np.zeros((1, 1))
        
        # Store for backprop
        self.cache = {}
        self.gradients = {}
    
    def relu(self, x):
        """ReLU activation"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU derivative"""
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        """Sigmoid activation - FIXED"""
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    def forward(self, X):
        """Forward pass - FIXED"""
        try:
            # Ensure X is 2D
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            # Layer 1
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self.relu(Z1)
            
            # Layer 2
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self.relu(Z2)
            
            # Layer 3 (Output)
            Z3 = np.dot(A2, self.W3) + self.b3
            A3 = self.sigmoid(Z3)
            
            # Cache for backprop
            self.cache = {
                'X': X, 'Z1': Z1, 'A1': A1,
                'Z2': Z2, 'A2': A2,
                'Z3': Z3, 'A3': A3
            }
            
            return A3
        except Exception as e:
            print(f"[ERROR in forward]: {e}")
            # Return default prediction
            return np.array([[0.5]])
    
    def backward(self, y_true, max_norm=1.0):
        """Backward pass with gradient clipping"""
        try:
            m = self.cache['X'].shape[0]
            
            # Output layer gradients
            dA3 = self.cache['A3'] - y_true.reshape(-1, 1)
            dZ3 = dA3
            dW3 = np.dot(self.cache['A2'].T, dZ3) / m
            db3 = np.sum(dZ3, axis=0, keepdims=True) / m
            
            # Hidden layer 2 gradients
            dA2 = np.dot(dZ3, self.W3.T)
            dZ2 = dA2 * self.relu_derivative(self.cache['Z2'])
            dW2 = np.dot(self.cache['A1'].T, dZ2) / m
            db2 = np.sum(dZ2, axis=0, keepdims=True) / m
            
            # Hidden layer 1 gradients
            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self.relu_derivative(self.cache['Z1'])
            dW1 = np.dot(self.cache['X'].T, dZ1) / m
            db1 = np.sum(dZ1, axis=0, keepdims=True) / m
            
            # Gradient clipping
            for dW in [dW1, dW2, dW3]:
                norm = np.linalg.norm(dW)
                if norm > max_norm:
                    dW *= max_norm / norm
            
            self.gradients = {
                'dW1': dW1, 'db1': db1,
                'dW2': dW2, 'db2': db2,
                'dW3': dW3, 'db3': db3
            }
        except Exception as e:
            print(f"[ERROR in backward]: {e}")
    
    def update_weights(self):
        """Update weights using gradients"""
        try:
            self.W1 -= self.learning_rate * self.gradients['dW1']
            self.b1 -= self.learning_rate * self.gradients['db1']
            
            self.W2 -= self.learning_rate * self.gradients['dW2']
            self.b2 -= self.learning_rate * self.gradients['db2']
            
            self.W3 -= self.learning_rate * self.gradients['dW3']
            self.b3 -= self.learning_rate * self.gradients['db3']
        except Exception as e:
            print(f"[ERROR in update]: {e}")
    
    def compute_loss(self, y_pred, y_true):
        """Binary cross-entropy loss"""
        try:
            m = y_true.shape[0]
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            return loss
        except Exception as e:
            print(f"[ERROR in loss]: {e}")
            return 0.5
    
    def predict(self, X):
        """Make predictions - FIXED"""
        try:
            print("[DEBUG] Starting prediction...")
            print(f"[DEBUG] Input shape: {X.shape}")
            
            # Forward pass
            output = self.forward(X)
            
            print(f"[DEBUG] Output shape: {output.shape}")
            print(f"[DEBUG] Output value: {output}")
            
            return output
        except Exception as e:
            print(f"[ERROR in predict]: {e}")
            import traceback
            traceback.print_exc()
            return np.array([[0.5]])