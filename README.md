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

$$
y = K\theta^* + \varepsilon
$$

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
├── setup_2d.py / setup_3d.py
├── 2d_baseline.py
├── 2d_direct_solve.py
├── 2d_l2solve.py
├── 2d_regularization_solve.py
├── 3d_baseline.py
├── 3d_direct_solve.py
├── 3d_l2solve.py
├── 3d_regularization.py


---

## 🧪 Key Results

### ❌ Direct Least Squares
- Completely fails due to ill-conditioning  
- Noise is massively amplified  
- Structure is destroyed  

### ⚠️ L2 Regularization
- Stable solution  
- But overly smooth  
- Fails to recover sharp boundaries  

### ✅ TV Regularization (Best)
- Preserves sharp edges  
- Recovers piecewise-constant structure  
- Achieves best reconstruction accuracy  

---

## 📊 2D vs 3D Insights

### 2D Case
- TV accurately recovers block structures  
- L2 produces blurred results  

### 3D Extension
- TV successfully localizes heat sources  
- L2 fails to preserve structure  

---

## ⚠️ Important Insight

There is **no universal best method**:

| Scenario | Best Method |
|--------|-----------|
| Sharp / piecewise constant | TV |
| Smooth / continuous | L2 |

- TV may cause **staircasing artifacts**  
- L2 performs better for smooth signals  

---

## 🚀 How to Run

### Install dependencies

```bash
pip install numpy scipy matplotlib cvxpy
python 2d_baseline.py
python 2d_direct_solve.py
python 2d_l2solve.py
python 2d_regularization_solve.py
python 3d_baseline.py
python 3d_direct_solve.py
python 3d_l2solve.py
python 3d_regularization.py

