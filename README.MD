<!-- logo in the center -->
<div align="center">
<img src="img/oikan_logo.png" alt="OIKAN Logo" width="200"/>
</div>

# 30 Days of development OIKAN (v0.0.3) Library

[![Star](https://img.shields.io/github/stars/silvermete0r/30github.svg?logo=github&style=flat-square)](https://github.com/silvermete0r/30github)&nbsp;
[![Fork](https://img.shields.io/github/forks/silvermete0r/30github.svg?logo=github&style=flat-square)](https://github.com/silvermete0r/30github)&nbsp;
[![Watch](https://img.shields.io/github/watchers/silvermete0r/30github.svg?logo=github&style=flat-square)](https://github.com/silvermete0r/30github)&nbsp;
<a href="https://dataflow.kz/"><img src="https://img.shields.io/badge/Powered%20by-Dataflow-line.svg" alt="dataflow" /></a>&nbsp;

* **Programming Languages:** Python; 
* **Main goal:** Applying traditional neural networks in combination with symbolic regression approaches based on Kolmogorov-Arnold representation theorem to obtain more accurate interpretable symbolic representations with better performance.

* **Idea:** 

* **Duration:** 30 days (1 month)

* **Deadlines:**
  * **Final Pre-Defense:** 24 May, 2025
  * **Final Defense:** 9 - 13 June, 2025

## Table of Contents
- [Day 0: Intro to Kolmogorov Arnold Networks](#day-0-intro-to-kolmogorov-arnold-networks)
- [Day 1: Implementing KAN for Regression Problems in Python](#day-1-implementing-kan-for-regression-problems-in-python)
- [Day 2: Implementing KAN for Classification Problems in Python](#day-2-implementing-kan-for-classification-problems-in-python)


## Day 0: Intro to Kolmogorov Arnold Networks

### Introduction

Kolmogorov-Arnold Networks (KANs) are a recent innovation in neural network architecture, drawing upon the Kolmogorov-Arnold representation theorem. This theorem, developed by [Andrey Kolmogorov](https://en.wikipedia.org/wiki/Andrey_Kolmogorov) and [Vladimir Arnold](https://en.wikipedia.org/wiki/Vladimir_Arnold), demonstrates that any continuous multivariable function can be expressed through the superposition of a finite number of univariate functions. KANs replace the fixed linear weights of traditional neural networks with learnable univariate functions, enhancing flexibility and interpretability.

### Historical Context

The Kolmogorov-Arnold theorem, introduced by Andrey Kolmogorov in 1957, revolutionized the representation of continuous multivariable functions by breaking them down into simpler univariate functions. Vladimir Arnold further validated and extended Kolmogorov's work in 1958, proving its effectiveness in representing higher-dimensional functions1. Their work provided new methods for machine learning, data fitting, and solving partial differential equations1.

### Mathematical Formulation

![KART: Mathematical Formulation](img/math-formulation-kan.png)

### How KANs Work

KANs use the Kolmogorov-Arnold Representation (KAR) theorem alongside [B-splines](https://en.wikipedia.org/wiki/B-spline) to create a dynamic model. The KAR theorem decomposes complex functions into simpler ones, and KANs apply this principle to each edge within the network, making each edge a learnable B-spline activation function. During training, each B-spline adjusts its control points through backpropagation, refining its approach to data with each iteration and continuously improving accuracy and efficiency.

### How B-Splines Work

B-splines are piecewise polynomial functions that are smooth at the joints, making them ideal for approximating complex functions. They are defined by a set of control points that determine the shape of the curve. By adjusting these control points, the B-spline can adapt to various data patterns, providing a flexible and powerful tool for function approximation.

### Comparison with Traditional Neural Networks (MLPs)

![KAN vs. MLP in Original Paper](img/kan_mlp_comparison_table_orig.png)

*🔗 Source: [(Arxiv) KAN: Kolmogorov–Arnold Networks](https://arxiv.org/abs/2404.19756)*

![KAN vs. MLP more fair comparison](img/kan_mlp_comparison_table_fair.png)

*🔗 Source: [(Arxiv) KAN or MLP: A Farier Comparison](https://arxiv.org/abs/2407.16674)*

### PyKAN Library Python

The PyKAN library provides a Python implementation of the KAN algorithm, allowing users to create, train, and evaluate KAN models. The library simplifies the process of working with KANs, providing a user-friendly interface for researchers and developers. PyKAN enables users to harness the power of KANs for various applications, from data fitting to machine learning tasks.

## Day 1: Implementing KAN for Regression Problems in Python

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

## Day 2: Implementing KAN for Classification Problems in Python

Today, we'll explore how to implement a KAN model in Python using the PyKAN library for the Titanic Survival Prediction Competition on Kaggle.

My implementation of PyKAN model for the Binary Classification Problem: ["Titanic - Machine Learning from Disaster"](https://www.kaggle.com/competitions/titanic) Kaggle Competition: [notebooks/titanic-pykan-predictions.ipynb](notebooks/titanic-pykan-predictions.ipynb)

## Day 3: Kolmogorov–Arnold-Informed neural network: A physics-informed deep learning framework for solving PDEs based on Kolmogorov–Arnold Networks

Today, we'll explore how to implement a Kolmogorov–Arnold-Informed neural network (KAINN) for solving partial differential equations (PDEs) based on Kolmogorov–Arnold Networks.

* The paper introduces the Kolmogorov–Arnold-Informed Neural Network (KINN), a physics‐informed deep learning framework that uses Kolmogorov–Arnold Networks (KANs) instead of conventional multilayer perceptrons (MLPs). KANs are noted for their improved interpretability and require fewer parameters than standard MLPs.
* Unlike traditional approaches that rely on a single formulation, the framework considers various PDE formulations (strong form, energy form, inverse form). Although mathematically equivalent, these forms lead to different computational behaviors, and KINN is designed to effectively handle these variations.
* Extensive numerical experiments—including multi-scale problems, singularities, stress concentration, nonlinear hyperelasticity, and heterogeneous materials—demonstrate that KINN achieves higher accuracy and faster convergence compared to traditional PINNs based on MLPs. An exception was noted in some complex geometry problems where performance gains were less pronounced.
* The framework builds on the Kolmogorov–Arnold representation theorem, which provides a theoretical foundation for the KAN architecture, thereby justifying its efficiency and reliability in approximating solutions to PDEs.
* The proposed method not only demonstrates significant improvements in solving forward and inverse PDE problems in computational solid mechanics but also comes with open-source code, ensuring that the results are reproducible and can serve as a valuable resource for further research in scientific computing.

**Practice Link:** *[Example 6: Solving Partial Differential Equation (PDE)](https://kindxiaoming.github.io/pykan/Example/Example_6_PDE_interpretation.html)*

**🔗 Source:** [Kolmogorov Arnold Informed neural network: A physics-informed deep learning framework for solving PDEs based on Kolmogorov Arnold Networks](https://arxiv.org/abs/2406.11045v1)

 

## Contributing to the OIKAN Library

If you're interested in contributing to the OIKAN library, feel free to check out the [Github repository](https://github.com/silvermete0r/oikan)

## References

1. [(Arxiv) KAN: Kolmogorov–Arnold Networks](https://arxiv.org/abs/2404.19756)
2. [(Arxiv) KAN 2.0: Kolmogorov-Arnold Networks Meet Science](https://arxiv.org/abs/2408.10205)
3. [(Arxiv) KAN or MLP: A Farier Comparison](https://arxiv.org/abs/2407.16674)
4. [(Arxiv) Rethinking the Function of Neurons in KANs](https://arxiv.org/abs/2407.20667)
5. [(Github) PyKAN Library Python](https://github.com/KindXiaoming/pykan)
6. [(Article)The Math Behind KAN – Kolmogorov-Arnold Networks](https://towardsdatascience.com/the-math-behind-kan-kolmogorov-arnold-networks-7c12a164ba95/)
7. [(Github) Awesome KAN(Kolmogorov-Arnold Network)](https://github.com/mintisan/awesome-kan)
8. [(Datacamp Tutorial) Kolmogorov-Arnold Networks (KANs): A Guide With Implementation](https://www.datacamp.com/tutorial/kolmogorov-arnold-networks)
9. [(Arxiv) Kolmogorov Arnold Informed neural network: A physics-informed deep learning framework for solving PDEs based on Kolmogorov Arnold Networks](https://arxiv.org/abs/2406.11045v1)
10. [(Arxiv) BSRBF-KAN: A combination of B-splines and Radial Basis Functions in Kolmogorov-Arnold Networks](https://arxiv.org/abs/2406.11173v1)
11. [(Arxiv) MLP-KAN: Unifying Deep Representation and Function Learning](https://arxiv.org/abs/2410.03027v1)
12. [(Github) An Efficient Implementation of Kolmogorov-Arnold Network](https://github.com/Blealtan/efficient-kan)
13. [(Arxiv) Neural Additive Models: Interpretable Machine Learning with Neural Nets](https://arxiv.org/abs/2004.13912)
14. [(Arxiv) Symbolic Regression is NP-hard](https://arxiv.org/pdf/2207.01018)