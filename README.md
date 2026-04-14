# Fake Review Detector 

A beginner-friendly deep learning project that detects fake reviews using an Artificial Neural Network with explainability features.

## Features

**Binary Classification with Confidence**: Get predictions with confidence scores (0-100%)  
**Suspicious Phrase Highlighting**: Identify which phrases influenced the prediction  
**Custom Training Loop**: Learn backpropagation with gradient clipping  
**Real Messy Data**: Trained on Amazon/Yelp review dataset  
**Gradient-Based Explainability**: See exactly why a review is flagged as fake

## Project Structure

```
fake-review-detector/
├── data_loader.py              # Dataset loading & preprocessing
├── feature_extractor.py        # TF-IDF vectorization
├── ann_model.py                # 3-layer ANN with custom backprop
├── explainer.py                # Gradient-based attribution
├── train.py                    # Training pipeline
├── inference.py                # Interactive demo
├── requirements.txt            # Dependencies
├── logs/                       # Visualizations
└── README.md                   # This file
```

## Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/Lovishkasoni/fake-review-detector.git
cd fake-review-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python train.py

# 4. Run interactive demo
python inference.py
```

## How It Works

### 1. Training

```bash
python train.py
```

This will:
1. Load Amazon reviews dataset
2. Extract TF-IDF features
3. Train 3-layer ANN with gradient clipping
4. Generate visualizations in `logs/`

### 2. Making Predictions

```bash
python inference.py
```

Example interaction:
```
Paste a review (or 'quit' to exit):
> Best product ever!!! Highly recommend!!! Must buy now!!!

=======================
PREDICTION RESULTS
=======================
Fake Probability: 76%
Confidence Score: 88%
Classification: LIKELY FAKE

Top Suspicious Phrases:
  1. 'Best product ever' → impact: +0.34
  2. 'Highly recommend' → impact: +0.29
  3. 'Must buy' → impact: +0.22
=======================
```

## Technical Details

### Model Architecture

```
Input (5000 features)
    ↓
Dense Layer (128 neurons, ReLU)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Output Layer (1 neuron, Sigmoid)
    ↓
Fake/Real Prediction [0, 1]
```

### Key Concepts

**Backpropagation**: Implemented from scratch to understand gradient flow  
**Gradient Clipping**: Prevents exploding gradients during training  
**TF-IDF Vectorization**: Converts text to numerical features  
**Gradient Attribution**: Maps gradients back to original phrases for explainability  

### Training Details

* **Dataset**: 1000 reviews (balanced fake/real)
* **Epochs**: 50
* **Batch Size**: 32
* **Learning Rate**: 0.01
* **Optimizer**: Custom SGD with gradient clipping (max_norm=1.0)

## Learning Outcomes

* Implement backpropagation manually  
* Handle gradient explosion with clipping  
* Apply NLP fundamentals (TF-IDF, tokenization)  
* Understand gradient-based explainability  
* Work with real messy text data  
* Bridge deep learning theory with practice

## Expected Results

* **Accuracy**: 75-85% on test set
* **Inference Time**: <100ms per review
* **Explainability**: Top phrases clearly indicate fake reviews

## File Descriptions

| File | Purpose |
|------|---------|
| `data_loader.py` | Loads Amazon reviews, handles preprocessing |
| `feature_extractor.py` | TF-IDF vectorization with phrase mapping |
| `ann_model.py` | 3-layer ANN, forward/backward pass |
| `explainer.py` | Gradient-based phrase attribution |
| `train.py` | Training loop, logging, visualization |
| `inference.py` | Interactive demo for testing |

## Troubleshooting

**Model not converging?**  
→ Try adjusting learning_rate or increasing epochs

**Memory issues?**  
→ Reduce max_features in FeatureExtractor or batch_size in train.py

**Predictions not making sense?**  
→ Ensure model_weights.npy exists (run train.py first)

## References

* Backpropagation: [3Blue1Brown Neural Networks Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
* Gradient-based Explanations: [Integrated Gradients Paper](https://arxiv.org/abs/1703.01365)
* TF-IDF: [scikit-learn Documentation](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

## License

MIT License - Feel free to use for learning!
