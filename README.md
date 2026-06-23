# Foundational Machine Learning Models 

**Overview:** A collection of core machine learning algorithms implemented entirely from scratch in pure Python and NumPy, alongside their PyTorch equivalents. 

As a first-year engineering student, I built this repository as a personal learning project to "look under the hood" of modern AI. Instead of relying on high level APIs like `scikit-learn` to simply call `.fit()`, the goal of this project was to manually engineer the underlying mathematics—specifically loss functions, forward passes, and gradient descent—to build a deep intuition for how neural networks and regression models actually learn.

---

## Repository Structure

This repository is divided into manual implementations and framework-optimized benchmarks:

* **`/Linear Regression`**
  * A pure Python/NumPy implementation of Simple and Multiple Linear Regression.
  * *Focus:* Manual calculation of Mean Squared Error (MSE) and step by step gradient descent weight updates without auto differentiation.
* **`/Logistic Regression`**
  * A pure Python/NumPy implementation of Logistic Regression for binary classification.
  * *Focus:* Implementing the Sigmoid activation function from scratch.
* **`/Linear Regression with pytorch`**
  * The exact same mathematical models re-engineered using PyTorch tensors.
  * *Focus:* Benchmarking the execution performance and exploring PyTorch's native `autograd`computational graphs versus my manual math loops.

---
## Current status
Learning more models to implement from scratch. I will add them here soon.


