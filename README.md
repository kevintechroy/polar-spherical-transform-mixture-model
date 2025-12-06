# Polar-Spherical Transform Mixture Model (PSTMM)

A novel unsupervised machine learning model that leverages geometric transformational embeddings—specifically pairwise polar and ternary spherical transforms—to improve clustering and classification performance on complex datasets.

## 📌 Overview

The **Polar-Spherical Transform Mixture Model (PSTMM)** is a custom classifier that extends traditional Gaussian Mixture Models (GMM) by transforming input data into polar and spherical coordinate spaces before modeling. This approach allows the model to capture angular and radial relationships in data that standard Cartesian-based models often miss.

Key features include:
*   **Geometric Embeddings:** Utilizes pairwise polar and ternary spherical transforms to project data into new feature spaces.
*   **Hybrid Modeling:** Combines Gaussian distributions for magnitude (norm) components and von Mises distributions for angular components.
*   **Scikit-Learn Compatibility:** Fully compatible `BaseEstimator` and `ClassifierMixin` wrapper (`PolarSphericalClassifier`) for seamless integration with `sklearn` pipelines, `GridSearchCV`, and cross-validation.
*   **Robust Initialization:** Implements Expectation-Maximization (EM) with multiple initializations to ensure convergence to optimal parameters.




