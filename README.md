# fashion_minst-NN-model
# Fashion-MNIST Classification with Keras

This notebook demonstrates a simple neural network implementation for classifying images from the **Fashion-MNIST** dataset using TensorFlow/Keras.

## 📌 Overview
- **Dataset**: Fashion-MNIST (70,000 grayscale images of 10 fashion categories).
- **Model**: Sequential neural network with:
  - Input layer (flattened 28×28 images)
  - 3 hidden Dense layers (ReLU activation)
  - Output layer (Softmax for 10-class classification)
- **Optimizer**: Adam (with mini-batch gradient descent via `batch_size=32`).
- **Validation**: Accuracy ~89% on test data after 50 epochs.

## 🛠️ Key Steps
1. **Data Loading & Preprocessing**:
   - Normalized pixel values to `[0, 1]`.
2. **Model Architecture**:
   ```python
   model = Sequential([
       Flatten(input_shape=(28, 28)),
       Dense(132, activation='relu'),
       Dense(64, activation='relu'),
       Dense(32, activation='relu'),
       Dense(10, activation='softmax')
   ])
   ## 📊 Results
- **Training Accuracy**: ~96%  
- **Validation Accuracy**: ~89%  
*(See notebook for detailed epoch-wise logs.)*
