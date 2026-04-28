# 🔥 Inverse Heat Conduction via Convex Optimization

> Recovering sharp initial temperature distributions from noisy observations using convex optimization

📄 Full Report: See project PDF

---

## 📌 Overview

This project studies the **inverse heat conduction problem**, where the goal is to recover an unknown initial temperature field from noisy observations after heat diffusion.

The problem is **ill-posed** because:

- Heat diffusion is a strong smoothing process  
- High-frequency information is lost  
- Noise is significantly amplified during inversion  

To address this, we reformulate the problem as a **convex optimization problem with regularization**.

We compare three methods:

- Direct Least Squares (baseline)
- L2 (Tikhonov) Regularization
- Total Variation (TV) Regularization

---

## 🧠 Mathematical Formulation

We model the observation as:

\[
y = K\theta^* + \varepsilon
\]

Where:

- \( \theta^* \): true initial temperature field  
- \( K \): heat diffusion operator (Gaussian blur)  
- \( \varepsilon \): noise  

### Optimization Problems

**1. Least Squares**
\[
\min_\theta \|K\theta - y\|_2^2
\]

**2. L2 Regularization**
\[
\min_\theta \|K\theta - y\|_2^2 + \lambda \|\theta\|_2^2
\]

**3. Total Variation (TV)**
\[
\min_\theta \|K\theta - y\|_2^2 + \lambda \, TV(\theta)
\]

TV promotes **piecewise-constant structures**, allowing recovery of sharp edges.

---

## ⚙️ Project Structure
