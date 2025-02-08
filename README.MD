![KAN-30-Days-Challenge-Header](img/kan-research-30-days-challenge-header.png)

# 30 Days of Kolmogorov Arnold Networks Research

[![Star](https://img.shields.io/github/stars/silvermete0r/30github.svg?logo=github&style=flat-square)](https://github.com/silvermete0r/30github)&nbsp;
[![Fork](https://img.shields.io/github/forks/silvermete0r/30github.svg?logo=github&style=flat-square)](https://github.com/silvermete0r/30github)&nbsp;
[![Watch](https://img.shields.io/github/watchers/silvermete0r/30github.svg?logo=github&style=flat-square)](https://github.com/silvermete0r/30github)&nbsp;

* **Programming Languages:** Python, C++ / C; 
* **Main goal:** Identify effective ways to optimise KAN neural network processes (training, inference, visualisation ~ explainability) and find real-world applications of algorithm optimised for specific tasks.
* **Tasks:** 
  * **Research:** Study the KAN algorithm and its applications in various fields.
  * **Development:** Implement the KAN algorithm in Python and C++ / C using various methods.
  * **Optimisation:** Optimise the KAN algorithm for specific tasks.
  * **Applications:** Find real-world applications of the KAN algorithm.
  * **Visualisation:** Visualise the KAN algorithm for explainability.
* **Duration:** 30 days (1 month)

## Table of Contents
- [Day 0: Intro to Kolmogorov Arnold Networks](#day-0-intro-to-kolmogorov-arnold-networks)
- [References](#references)

## Day 0: Intro to Kolmogorov Arnold Networks

### Introduction

Kolmogorov-Arnold Networks (KANs) are a recent innovation in neural network architecture, drawing upon the Kolmogorov-Arnold representation theorem. This theorem, developed by [Andrey Kolmogorov](https://en.wikipedia.org/wiki/Andrey_Kolmogorov) and [Vladimir Arnold](https://en.wikipedia.org/wiki/Vladimir_Arnold), demonstrates that any continuous multivariable function can be expressed through the superposition of a finite number of univariate functions. KANs replace the fixed linear weights of traditional neural networks with learnable univariate functions, enhancing flexibility and interpretability.

### Historical Context

The Kolmogorov-Arnold theorem, introduced by Andrey Kolmogorov in 1957, revolutionized the representation of continuous multivariable functions by breaking them down into simpler univariate functions. Vladimir Arnold further validated and extended Kolmogorov's work in 1958, proving its effectiveness in representing higher-dimensional functions1. Their work provided new methods for machine learning, data fitting, and solving partial differential equations1.

### Mathematical Formulation

![KART: Mathematical Formulation](img/math-formulation-kan.png)

### How KANs Work

KANs use the Kolmogorov-Arnold Representation (KAR) theorem alongside [B-splines](https://en.wikipedia.org/wiki/B-spline) to create a dynamic model. The KAR theorem decomposes complex functions into simpler ones, and KANs apply this principle to each edge within the network, making each edge a learnable B-spline activation function. During training, each B-spline adjusts its control points through backpropagation, refining its approach to data with each iteration and continuously improving accuracy and efficiency.

### PyKAN Library Python

The PyKAN library provides a Python implementation of the KAN algorithm, allowing users to create, train, and evaluate KAN models. The library simplifies the process of working with KANs, providing a user-friendly interface for researchers and developers. PyKAN enables users to harness the power of KANs for various applications, from data fitting to machine learning tasks.

My implementation of PyKAN model for the Binary Classification Problem: ["Titanic - Machine Learning from Disaster"](https://www.kaggle.com/competitions/titanic) Kaggle Competition: [notebooks/titanic-pykan-predictions.ipynb](notebooks/titanic-pykan-predictions.ipynb)


*🔗 Source: [(Arxiv) KAN: Kolmogorov–Arnold Networks](https://arxiv.org/abs/2404.19756)*

## Day 1: Implementing KAN in Python

[PyKAN Library Python](https://github.com/KindXiaoming/pykan)

Today, we'll explore how to implement a KAN model in Python using the PyKAN library for the Housing Prices Prediction Competition on Kaggle.

**1.1. Installation:**

```bash
pip install pykan
```

**1.2. Initializing a KAN:**

```python
from kan import *

model = KAN(width=[X_train.shape[1], 20, 10, 1], grid=3, k=3, seed=42, device='cpu')
```

**Explanation of parameters:**

* `width=[X_train.shape[1], 20, 10, 1]`: Defines a network with input features, two hidden layers (20 and 10 neurons), and one output.
* `grid=3`: Sets the number of grid points for piecewise function representation.
* `k=3`: Defines the order of the spline function.
* `seed=42`: Ensures reproducibility.
* `device='cpu'`: Uses CPU for training (set 'cuda' if using a GPU).

**1.3. Training the KAN:**

```python
train_data = {
    'train_input': torch.tensor(X_train, dtype=torch.float64),
    'train_label': torch.tensor(y_train, dtype=torch.float64).view(-1, 1),
    'test_input': torch.tensor(X_val, dtype=torch.float64),
    'test_label': torch.tensor(y_val, dtype=torch.float64).view(-1, 1)
}

model.fit(train_data, opt="LBFGS", steps=100, lamb=0.001)
```

**Explanation of training parameters:**

* `opt="LBFGS"`: Uses the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) optimizer. Why? It's efficient for small datasets.
* `steps=100`: Sets the number of optimization steps.
* `lamb=0.001`: Defines the regularization parameter.

**1.4. Extracting Symbolic Formula**

*A key advantage of KAN is that it allows us to obtain a symbolic representation of the trained model. We can either manually set functions or let PyKAN automatically discover symbolic expressions.*

**1.4.1 Manual Mode**

```python
model.fix_symbolic(0, 0, 0, 'sin')  # First layer, first neuron, first function -> sin
model.fix_symbolic(0, 1, 0, 'x^2')  # First layer, second neuron, first function -> x^2
model.fix_symbolic(1, 0, 0, 'exp')  # Second layer, first neuron, first function -> exp
```

**1.4.2 Automatic Mode**

***PyKAN automatically choose the best-fitting symbolic functions from a predefined library***

```python
library = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']
model.auto_symbolic(lib=library)
```

**1.4.3. Printing Final Mathematical Representation of KAN model**

```python
from kan.utils import ex_round
final_formula = ex_round(model.symbolic_formula()[0][0], 4)
print("Final Model Formula:", final_formula)
```

**1.5. Visualizing the Trained KAN Model**

*Visualizing math formula as a network graph.*

```python
model.plot()
```

**1.6. Making Predictions**

```python
test_preds = model(torch.tensor(test_data, dtype=torch.float64)).cpu().detach().numpy()
```

**1.7. Predictions Submission**

```python
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': test_preds.flatten()})
submission.to_csv('submission.csv', index=False)
```

*KAN offers interpretable models with a strong mathematical foundation, making it a unique alternative to traditional deep learning architectures. Tomorrow, we'll explore hyperparameter tuning and advanced visualization techniques. 🚀*

My implementation of PyKAN model for the Regression Problem: ["Housing Prices"](https://www.kaggle.com/competitions/home-data-for-ml-course) Kaggle Competition: [notebooks/pykan-housing-prices-prediction.ipynb](notebooks/pykan-housing-prices-prediction.ipynb)

## References

1. [Kolmogorov-Arnold representation theorem](https://en.wikipedia.org/wiki/Kolmogorov-Arnold_representation_theorem)
2. [(Arxiv) KAN: Kolmogorov–Arnold Networks](https://arxiv.org/abs/2404.19756)
3. [PyKAN Library Python](https://github.com/KindXiaoming/pykan)