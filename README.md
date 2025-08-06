📦 CNN from Scratch (NumPy + MNIST)

This project implements a **Convolutional Neural Network (CNN)** entirely from scratch using only **NumPy**, without using any deep learning frameworks like TensorFlow or PyTorch.

It is trained on the **MNIST handwritten digits dataset** and achieves over **83% test accuracy** after just 3 epochs on a 1000-image subset. This project demonstrates a fundamental understanding of how CNNs work internally — including forward pass, backpropagation, and training logic.

---

## 📚 Table of Contents

- [🔍 Project Highlights](#-project-highlights)
- [🧠 Network Architecture](#-network-architecture)
- [📊 Performance](#-performance)
- [💻 Run Locally](#-run-locally)
- [🛠️ Possible Improvements](#-possible-improvements)
- [🎓 Learning Outcomes](#-learning-outcomes)
- [📌 Credits](#-credits)
- [📬 Contact](#-contact)
- [🏷️ License](#-license)

---

## 🔍 Project Highlights

- ✅ Custom implementation of:
  - `Conv3x3` convolutional layer
  - `MaxPool2` max-pooling layer
  - `Softmax` classifier
- ✅ Manual backpropagation for all layers
- ✅ Cross-entropy loss + SGD training
- ✅ Data normalization and reshaping
- ✅ Trained on a 1000-sample subset of MNIST for demonstration speed
- ✅ Epoch-wise and batch-wise logging of loss and accuracy

---

## 🧠 Network Architecture

```text
Input:       28x28 grayscale image
↓ Conv3x3    (8 filters)       → Output: 26x26x8
↓ MaxPool2   (2x2 pooling)     → Output: 13x13x8
↓ Flatten
↓ Softmax    (Fully connected) → Output: 10 class scores
```
---
💻 Run Locally

1. Clone the repository
  bash
```text
git clone https://github.com/yourusername/cnn-from-scratch.git
cd cnn-from-scratch
```
2. (Optional) Create and activate a virtual environment
  bash
```text
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```
3. Install dependencies
  bash
```text
pip install numpy tensorflow
```
4. Run the training script
  bash
```text
python cnn.py
```
You should see epoch-wise logs showing loss and accuracy.

🛠️ Possible Improvements

	•	Add support for multiple convolution + pooling layers
	•	Add dropout and batch normalization
	•	Implement additional optimizers (Adam, RMSprop)
	•	Train on the full MNIST dataset for better accuracy
	•	Make the code modular and dataset-agnostic (e.g., CIFAR-10)
	•	Visualize learned filters and activations
	•	Add a CLI or training dashboard for visualization
 
 🎓 Learning Outcomes
      
    This project helped me:
	
    •	Understand how CNNs work internally
  	•	Manually implement convolution, pooling, and softmax layers
  	•	Build the forward and backward pass logic from scratch
  	•	Train a neural network using raw NumPy
  	•	Appreciate the foundations of deep learning without using frameworks

⸻

📌 Credits

	•	MNIST dataset is provided via tensorflow.keras.datasets
	•	Inspired by multiple deep learning “from scratch” resources and coursework

⸻

📬 Contact

Feel free to reach out for suggestions, collaborations, or feedback:
	•	📧 Email: pandeydeepansh24@gmail.com
	•	🔗 LinkedIn: linkedin.com/in/deepansh-pandey-12235830a
