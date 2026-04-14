import numpy as np
import matplotlib.pyplot as plt
from data_loader import AmazonReviewLoader
from feature_extractor import FeatureExtractor
from ann_model import ANNModel
import os

def train():
    """Main training pipeline"""
    print("="*60)
    print("FAKE REVIEW DETECTOR - TRAINING")
    print("="*60)
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Load data
    print("\n[1/4] Loading data...")
    loader = AmazonReviewLoader()
    train_reviews, test_reviews, train_labels, test_labels = loader.load_data()
    
    # Extract features
    print("\n[2/4] Extracting features...")
    extractor = FeatureExtractor(max_features=5000)
    X_train = extractor.fit_transform(train_reviews)
    X_test = extractor.transform(test_reviews)
    
    feature_names = extractor.get_feature_names()
    print(f"Feature matrix shape: {X_train.shape}")
    
    # Initialize model
    print("\n[3/4] Initializing model...")
    input_dim = X_train.shape[1]
    model = ANNModel(input_dim=input_dim, hidden1=128, hidden2=64, learning_rate=0.01)
    print(f"Model created: {input_dim}")
    
    # Training loop
    print("\n[4/4] Training model...")
    epochs = 50
    batch_size = 32
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    num_batches = len(X_train) // batch_size
    
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = train_labels[indices]
        
        # Mini-batch training
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # Forward pass
            y_pred = model.forward(X_batch)
            
            # Compute loss
            loss = model.compute_loss(y_pred, y_batch)
            epoch_loss += loss
            
            # Backward pass
            model.backward(y_batch)
            
            # Update weights
            model.update_weights()
            
            # Accuracy
            predictions = (y_pred > 0.5).astype(int).flatten()
            correct += np.sum(predictions == y_batch)
        
        train_loss = epoch_loss / num_batches
        train_accuracy = correct / len(X_train)
        
        # Test accuracy
        y_test_pred = model.predict(X_test)
        test_pred = (y_test_pred > 0.5).astype(int).flatten()
        test_accuracy = np.sum(test_pred == test_labels) / len(test_labels)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f}")
    
    # Save model
    np.save('model_weights.npy', {
        'W1': model.W1, 'b1': model.b1,
        'W2': model.W2, 'b2': model.b2,
        'W3': model.W3, 'b3': model.b3
    })
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Loss curve
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    # Accuracy curve
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train')
    plt.plot(test_accuracies, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Confidence distribution
    plt.subplot(1, 3, 3)
    confidences = model.predict(X_test).flatten()
    plt.hist(confidences, bins=30, alpha=0.7)
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('logs/training_metrics.png', dpi=100)
    print("Saved: logs/training_metrics.png")
    plt.close()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.4f}")
    print("="*60)

if __name__ == "__main__":
    train()