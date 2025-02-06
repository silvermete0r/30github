![KAN-30-Days-Challenge-Header](https://sun9-52.userapi.com/impg/meyDkrQTZuDYjaQZHlSFsYx9TKbZ6Mq4W7mCPg/2tEA4Boh36Q.jpg?size=1405x433&quality=95&sign=7b73ae4421d44d34b0e32492a4735cf1&type=album)

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

### How KANs Work

KANs use the Kolmogorov-Arnold Representation (KAR) theorem alongside [B-splines](https://en.wikipedia.org/wiki/B-spline) to create a dynamic model. The KAR theorem decomposes complex functions into simpler ones, and KANs apply this principle to each edge within the network, making each edge a learnable B-spline activation function. During training, each B-spline adjusts its control points through backpropagation, refining its approach to data with each iteration and continuously improving accuracy and efficiency.


*🔗 Source: [(Arxiv) KAN: Kolmogorov–Arnold Networks](https://arxiv.org/abs/2404.19756)*


## References

1. [Kolmogorov-Arnold representation theorem](https://en.wikipedia.org/wiki/Kolmogorov-Arnold_representation_theorem)
2. [(Arxiv) KAN: Kolmogorov–Arnold Networks](https://arxiv.org/abs/2404.19756)