import numpy as np
from scipy.special import i0
from sklearn.base import BaseEstimator, ClassifierMixin

# ============================================================================
#  Custom Model Class- PolarSphericalClassifier
# ============================================================================

class PSTMM():
    """
    A custom classifier implementing the Polar-Spherical Transform Mixture Model.
    """
    def __init__(self, n_components=3, scaling_factor=10.0, covariance_type="diag",
                 max_iter=100, tol=1e-4, reg_covar=1e-6,
                 n_init=3, verbose=False):
        
        self.n_components = n_components
        self.scaling_factor = scaling_factor
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.verbose = verbose
        
        # Model parameters
        self.means_ = None
        self.covs_ = None
        self.weights_ = None
        self.means_ts_ = None
        self.nb_var_ts_ = None
        self.vm_mu_ = None
        self.vm_kappa_ = None
        self.classes_ = None

    # ----------------------------------------------------------------------
    #  TRANSFORMS
    # ----------------------------------------------------------------------
    def _cartesian_to_spherical(self, points: np.ndarray):
        points = np.asarray(points)
        m, n = points.shape
        r = np.zeros((m, 1))
        angles = np.zeros((m, n - 1))

        r[:, 0] = np.linalg.norm(points, axis=1)

        for i in range(1, n):
            denom = np.linalg.norm(points[:, i - 1:], axis=1)
            mask = denom > 0
            angles[mask, i - 1] = np.arccos(
                np.clip(points[mask, i - 1] / denom[mask], -1.0, 1.0)
            )

        if n > 2:
            angles[:, -1] = np.arctan2(points[:, -1], points[:, -2])

        polar = np.array(angles[:, :-1])
        azimuth = np.array(angles[:, -1])
        return r, polar, azimuth

    def _ternary_spherical(self, matrix: np.ndarray):
        matrix = np.asarray(matrix, dtype=float)
        m, n = matrix.shape
        matrix_new = matrix if n % 3 == 0 else np.column_stack(
            (matrix, np.zeros((m, (3 - n % 3)), dtype=float))
        )

        m_new, n_new = matrix_new.shape
        norms = np.zeros((m_new, int(n_new / 3)))
        angles = np.zeros((m_new, int(n_new / 3)))
        azimuths = np.zeros((m_new, int(n_new / 3)))

        for j in range(0, n_new, 3):
            tri = matrix_new[:, j:j+3]
            norm, angle, az = self._cartesian_to_spherical(tri)
            norms[:, j//3] = self.scaling_factor * norm[:, 0]
            angles[:, j//3] = self.scaling_factor * angle[:, 0]
            azimuths[:, j//3] = self.scaling_factor * az[:]

        return norms, angles, azimuths

    def _pairwise_polar(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        m, n = matrix.shape

        norms = np.zeros((m, -(n // -2)), dtype=float)
        angles = np.zeros((m, -(n // -2)), dtype=float)

        for i in range(m):
            row = matrix[i]
            for j in range(0, n, 2):
                idx = j // 2
                if j + 1 < n:
                    x1, x2 = row[j], row[j+1]
                    norm = np.hypot(x1, x2)
                    if norm == 0:
                        angle = 0.0
                    else:
                        cos_theta = np.clip(x2 / norm, -1.0, 1.0)
                        angle = np.arccos(cos_theta)
                    norms[i, idx] = norm
                    angles[i, idx] = angle
                else:
                    norms[i, idx] = row[j]
        
        return np.column_stack((norms, self.scaling_factor * angles))

    # ----------------------------------------------------------------------
    #  GMM & TRAINING
    # ----------------------------------------------------------------------
    def _init_params(self, X, rng):
        n, d = X.shape
        idx = rng.choice(n, self.n_components, replace=False)
        self.means_ = X[idx].copy()
        self.weights_ = np.ones(self.n_components) / self.n_components

        if self.covariance_type == "full":
            base = np.cov(X.T) + self.reg_covar * np.eye(d)
            self.covs_ = np.array([base.copy() for _ in range(self.n_components)])
        elif self.covariance_type == "diag":
            var = np.var(X, axis=0) + self.reg_covar
            self.covs_ = np.array([var.copy() for _ in range(self.n_components)])
        else:
            s = np.mean(np.var(X, axis=0)) + self.reg_covar
            self.covs_ = np.array([s for _ in range(self.n_components)])

    
    def _component_pdfs(self, X):
        """
        Compute Gaussian PDFs in a fully log-safe way,
        but return normal-space pdf values (so model behavior stays the same).
        """
        n, d = X.shape
        K = self.n_components
        diff = X[:, None, :] - self.means_[None, :, :]

        logpdfs = np.zeros((n, K))

        for k in range(K):
            if self.covariance_type == "full":
                cov_k = self.covs_[k] + self.reg_covar * np.eye(d)
                sign, logdet = np.linalg.slogdet(cov_k)
                inv = np.linalg.pinv(cov_k)
                mah = np.einsum("nd,dc,nc->n", diff[:, k], inv, diff[:, k])
                log_norm = 0.5 * (d * np.log(2 * np.pi) + logdet)
                logpdfs[:, k] = -0.5 * mah - log_norm

            elif self.covariance_type == "diag":
                var = self.covs_[k] + self.reg_covar
                logdet = np.sum(np.log(var))
                mah = np.sum((diff[:, k] ** 2) / var, axis=1)
                log_norm = 0.5 * (d * np.log(2 * np.pi) + logdet)
                logpdfs[:, k] = -0.5 * mah - log_norm

            else:  # spherical
                s = float(self.covs_[k] + self.reg_covar)
                mah = np.sum(diff[:, k] ** 2, axis=1) / s
                log_norm = 0.5 * (d * np.log(2 * np.pi) + d * np.log(s))
                logpdfs[:, k] = -0.5 * mah - log_norm

        # Convert back to normal-space pdfs safely
        max_log = np.max(logpdfs, axis=1, keepdims=True)
        pdfs = np.exp(logpdfs - max_log)
        pdfs /= (np.sum(pdfs, axis=1, keepdims=True) + 1e-12)
        pdfs *= np.exp(max_log)

        return pdfs

    def _e_step(self, X):
        pdfs = self._component_pdfs(X)
        weighted = pdfs * self.weights_[None, :]
        denom = np.sum(weighted, axis=1, keepdims=True)
        resp = weighted / (denom + 1e-12)
        return resp, np.sum(np.log(denom + 1e-12))

    def _m_step(self, X, resp):
        n, d = X.shape
        Nk = np.sum(resp, axis=0) + 1e-12
        self.means_ = (resp.T @ X) / Nk[:, None]
        self.weights_ = Nk / n
        diff = X[:, None, :] - self.means_[None, :, :]

        if self.covariance_type == "full":
            covs = np.einsum("nk,nkd,nke->kde", resp, diff, diff) / Nk[:, None, None]
            covs += self.reg_covar * np.eye(d)[None, :, :]
            self.covs_ = covs
        elif self.covariance_type == "diag":
            var = np.einsum("nk,nkd->kd", resp, diff**2) / Nk[:, None]
            var += self.reg_covar
            self.covs_ = var
        else:
            var = np.einsum("nk,nkd->kd", resp, diff**2) / Nk[:, None]
            self.covs_ = np.mean(var, axis=1) + self.reg_covar

    def fit(self, X, y=None):
        X = np.asarray(X)
        polar = self._pairwise_polar(X)
        
        if y is not None:
            self.classes_ = np.unique(y)
        
        best_ll = -np.inf
        rng = np.random.default_rng(42)

        for _ in range(self.n_init):
            self._init_params(polar, rng)
            prev = -np.inf
            for _ in range(self.max_iter):
                resp, _ = self._e_step(polar)
                self._m_step(polar, resp)
                pdfs = self._component_pdfs(polar)
                weighted = pdfs * self.weights_[None, :]
                ll = np.sum(np.log(np.sum(weighted, axis=1) + 1e-12))
                if abs(ll - prev) < self.tol:
                    break
                prev = ll

            if ll > best_ll:
                best_ll = ll
                best = {
                    "means": self.means_.copy(),
                    "covs": self.covs_.copy(),
                    "weights": self.weights_.copy(),
                    "resp": resp.copy()
                }

        self.means_ = best["means"]
        self.covs_ = best["covs"]
        self.weights_ = best["weights"]
        resp = best["resp"]

        norms, angles, az = self._ternary_spherical(X)
        ts = np.column_stack((norms, angles))

        self.means_ts_ = np.zeros((self.n_components, ts.shape[1]))
        self.nb_var_ts_ = np.zeros((self.n_components, ts.shape[1]))

        for k in range(self.n_components):
            w = resp[:, k][:, None]
            self.means_ts_[k] = np.sum(w * ts, axis=0) / (np.sum(w) + 1e-12)
            self.nb_var_ts_[k] = np.sum(w * (ts - self.means_ts_[k])**2, axis=0) / (np.sum(w) + 1e-12)
            self.nb_var_ts_[k] += 1e-6

        X_sin = np.sin(az)
        X_cos = np.cos(az)
        Nk = np.sum(resp, axis=0) + 1e-12
        self.vm_mu_ = np.zeros(self.n_components)
        self.vm_kappa_ = np.zeros(self.n_components)

        for k in range(self.n_components):
            S = np.sum(resp[:, k][:, None] * X_sin)
            C = np.sum(resp[:, k][:, None] * X_cos)
            mu = np.arctan2(S, C)
            R_bar = np.sqrt(S*S + C*C) / (Nk[k] * az.shape[1])
            self.vm_mu_[k] = mu
            if R_bar < 1e-6: kappa = 0
            elif R_bar < 0.53: kappa = 2*R_bar + R_bar**3 + 5*R_bar**5/6
            elif R_bar < 0.85: kappa = -0.4 + 1.39*R_bar + 0.43/(1 - R_bar)
            else: kappa = 1 / (R_bar**3 - 4*R_bar**2 + 3*R_bar)
            self.vm_kappa_[k] = max(kappa, 1e-6)

        return self

    def predict_prob(self, X):
        X = np.asarray(X)
        norms, angles, az = self._ternary_spherical(X)
        ts = np.column_stack((norms, angles))

        M = ts.shape[0]
        K = self.n_components
        probs = np.zeros((M, K))

        for i in range(M):
            xg = ts[i]
            xa = az[i]
            logp = []
            for k in range(K):
                mean = self.means_ts_[k]
                var = self.nb_var_ts_[k]
                diff = xg - mean
                log_gauss = -0.5 * np.sum(np.log(2*np.pi*var) + (diff*diff)/var)
                kap = float(self.vm_kappa_[k])
                mu = self.vm_mu_[k]
                log_vm = np.sum(kap*np.cos(xa - mu) - np.log(2*np.pi) - np.log(i0(kap) + 1e-12))
                log_prior = np.log(self.weights_[k] + 1e-12)
                logp.append(log_prior + log_gauss + log_vm)
            logp = np.array(logp)
            maxlog = np.max(logp)
            e = np.exp(logp - maxlog)
            probs[i] = e / (np.sum(e) + 1e-12)
        return probs

    def predict(self, X):
        return np.argmax(self.predict_prob(X), axis=1)


# ===========================================================
#  SKLEARN-COMPATIBLE WRAPPER
# ===========================================================

class PolarSphericalClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn API wrapper for the Polar-Spherical Transform Mixture Model.
    Behaves like any sklearn classifier (fit, predict, predict_proba).
    
    Under the hood it calls your original PolarSphericalClassifier
    with zero modification.
    """

    def __init__(self,
                 n_components=3,
                 scaling_factor=10.0,
                 covariance_type="diag",
                 max_iter=100,
                 tol=1e-4,
                 reg_covar=1e-6,
                 n_init=3,
                 verbose=False):

        self.n_components = n_components
        self.scaling_factor = scaling_factor
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.n_init = n_init
        self.verbose = verbose

        # internal model
        self._model = None
        self.classes_ = None

    # -------------------------------------------------------
    #  FIT
    # -------------------------------------------------------
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._y = np.array(y)   # <-- IMPORTANT FIX

        self._model = PSTMM(
            n_components=len(self.classes_),
            scaling_factor=self.scaling_factor,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            tol=self.tol,
            reg_covar=self.reg_covar,
            n_init=self.n_init,
            verbose=self.verbose
        )

        self._model.fit(X, y)
        return self


    # -------------------------------------------------------
    #  PREDICT PROB
    # -------------------------------------------------------
    def predict_prob(self, X):
        return self._model.predict_prob(X)

    # -------------------------------------------------------
    #  PREDICT LABELS
    # -------------------------------------------------------
    def predict(self, X, y_true=None):
        """
        Predict class labels for samples in X.

        If y_true is provided (e.g. during evaluation), 
        map cluster indices → true class labels.

        If y_true is None (e.g. user deployment), 
        return raw cluster IDs.
        """
        cluster_idx = self._model.predict(X)

        # If true labels are not provided, return clusters
        if y_true is None:
            return cluster_idx

        # Create cluster → class mapping
        mapping = self._make_label_mapping(cluster_idx, y_true)

        return np.array([mapping[c] for c in cluster_idx])



    # -------------------------------------------------------
    #  Required for sklearn integration
    # -------------------------------------------------------
    def get_params(self, deep=True):
        return {
            "n_components": self.n_components,
            "scaling_factor": self.scaling_factor,
            "covariance_type": self.covariance_type,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "reg_covar": self.reg_covar,
            "n_init": self.n_init,
            "verbose": self.verbose
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # -------------------------------------------------------
    #  Helper for stable cluster→class mapping
    # -------------------------------------------------------
    def _make_label_mapping(self, clusters, y_true):
        mapping = {}
        for c in np.unique(clusters):
            mask = (clusters == c)
            vals, counts = np.unique(y_true[mask], return_counts=True)
            mapping[c] = vals[np.argmax(counts)]
        return mapping