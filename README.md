# Twitter Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.0+-red.svg)](https://keras.io/)
[![NLTK](https://img.shields.io/badge/NLTK-Latest-green.svg)](https://www.nltk.org/)

## Project Overview
A deep learning project implementing advanced neural network architectures to classify tweet sentiments. The project showcases the complete deep learning pipeline, from text preprocessing and feature engineering to model training and evaluation. This project was developed as part of the KIT315 unit assessment, demonstrating practical applications of deep learning in natural language processing.

## Key Features
- Implementation of two deep learning architectures
- Advanced text preprocessing pipeline
- Word embeddings for text representation
- Comprehensive model evaluation and comparison
- Detailed text visualisation and analysis
- Hyperparameter optimisation

## Technologies Used
- Python 3.7+
- Jupyter Notebook
- Key Libraries:
  - TensorFlow & Keras: Deep learning models
  - NLTK: Natural language processing
  - pandas & numpy: Data manipulation
  - matplotlib & seaborn: Data visualisation
  - scikit-learn: Model evaluation

## Models Implemented
1. Bidirectional LSTM
2. CNN-LSTM Hybrid Model

## Project Pipeline

### Text Preprocessing
- Text cleaning and normalisation
- Stop word removal
- Tokenisation
- Sequence padding

### Model Development
- Word embeddings
- Deep learning architecture design
- Hyperparameter tuning
- Dropout for regularisation

### Model Evaluation
- Classification metrics
- Confusion matrix analysis
- Training history visualisation

## Results
The CNN-LSTM hybrid model demonstrated superior performance with:
- Improved accuracy across all sentiment categories
- Better handling of complex sentence structures
- More robust feature extraction

## Project Structure
```
twitter-sentiment-classifier/
│
├── twitter_sentiment_analysis_dl.ipynb   # Main Jupyter notebook
├── README.md                             # Project documentation
├── report/
│   └── technical_report.pdf              # Detailed technical report
└── data/                           
    ├── train.csv                         # Training dataset
    └── test.csv                          # Test dataset
```

## Setup and Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Eusha425/twitter-sentiment-classifier.git
   ```

2. Install required packages:
   ```python
   import pandas as pd
   import numpy as np
   import tensorflow as tf
   import nltk
   from tensorflow.keras.layers import *
   from sklearn.model_selection import train_test_split
   from tensorflow.keras.preprocessing.text import Tokenizer
   ```

3. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook twitter_sentiment_analysis_dl.ipynb
   ```

## Future Improvements
1. **Text Processing**
   - Advanced text cleaning techniques
   - Custom tokenisation methods
   - Handling of emojis and special characters

2. **Model Architecture**
   - Transformer-based models
   - Attention mechanisms
   - Pre-trained word embeddings

3. **Evaluation**
   - Cross-validation implementation
   - ROC curve analysis
   - Model interpretability

## References
1. Research papers and documentation referenced in the technical report
2. Deep learning architecture implementations
3. Natural language processing techniques


## Licence
This project is licenced under the MIT licence - see the [licence](https://github.com/Eusha425/twitter-sentiment-classifier/blob/main/LICENSE) file for details.
