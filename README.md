#  Handwritten Digit Recognition: EMNIST vs MNIST

This project is a deep learning application that compares predictions from two separate neural network models trained on the MNIST and EMNIST datasets. The application features an interactive GUI where users can draw digits and view real-time model predictions.

---

## ✨ Features

- ✅ Draw digits using your mouse in the GUI
- ✅ Predict with both MNIST and EMNIST models
- ✅ Real-time softmax confidence display
- ✅ Image preprocessing: centering, normalization, inversion
- ✅ Tkinter GUI with interactive canvas and control buttons

---

## 🔧 Installation

```
bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
pip install -r requirements.txt
```

Or install manually:
```
pip install torch torchvision pillow scipy numpy matplotlib
```

---

# 🚀 How to Run

```
python complete_project.py
```

  - Use your mouse to draw a digit on the canvas

  - Click Predict to view predictions from both models

  - Click Clear to reset

  - Click Quit to exit the app
---

# 📊 Model Architecture

Both models use a simple CNN architecture:

  - 2 Conv2D layers (ReLU + MaxPooling)

  - Dropout layers for regularization

  - Fully connected layer (128 units)

  - Output layer (10 units + softmax)

---
# 👥 Contributors

  - Ömer Kocabaş

  - Mustafa Aydın
  
---

# 📄 License

This project is for educational purposes.
