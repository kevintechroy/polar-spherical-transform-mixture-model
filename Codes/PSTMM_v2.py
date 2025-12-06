import numpy as np
from scipy.special import i0
from sklearn.base import BaseEstimator, ClassifierMixin

# ---------------------------------------------------------------------------
# Polar-Spherical Transform Mixture Model (PSTMM) - updated version
# - preserves the variable names and overall structure
# - fixes ternary-spherical indexing bug
# - provides pairwise polar, cartesian->spherical, and ternary-spherical tra nsforms
# - computes Cartesian Gaussian params from resp and evaluates Gaussians in Cartesian space
# ---------------------------------------------------------------------------

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
        
        # Model parameters (original)
        self.means_ = None
        self.covs_ = None
        self.weights_ = None
        self.means_ts_ = None
        self.nb_var_ts_ = None
        self.vm_mu_ = None
        self.vm_kappa_ = None
        self.classes_ = None

        # New: Cartesian Gaussian params computed after EM using resp
        self.means_cart_ = None
        self.covs_cart_ = None

    # ----------------------------------------------------------------------
    #  TRANSFORMS
    # ----------------------------------------------------------------------
    def _cartesian_to_pairwise_polar(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        m, n = matrix.shape

        result = np.zeros((m, n), dtype=float)

        for i in range(m):
            row = matrix[i]
            for j in range(0, n, 2):
                if j + 1 < n:
                    x1, x2 = row[j], row[j+1]
                    norm = np.hypot(x1, x2)
                    if norm == 0:
                        angle = 0.0
                    else:
                        cos_theta = np.clip(x2 / norm, -1.0, 1.0)
                        angle = np.arccos(cos_theta)
                    result[i, j] = norm
                    result[i, j + 1] = angle
                else:
                    result[i, j] = row[j]
        
        return result

    def _cartesian_to_spherical(self, points: np.ndarray):
        points = np.asarray(points)
        m, n = points.shape
        r = np.zeros((m, 1))
        angles = np.zeros((m, max(0, n - 1)))

        r[:, 0] = np.linalg.norm(points, axis=1)

        for i in range(1, n):
            denom = np.linalg.norm(points[:, i - 1:], axis=1)
            mask = denom > 0
            angles[mask, i - 1] = np.arccos(
                np.clip(points[mask, i - 1] / denom[mask], -1.0, 1.0)
            )

        if n > 2:
            angles[:, -1] = np.arctan2(points[:, -1], points[:, -2])

        # polar: all angles except the last (if present)
        if angles.shape[1] > 1:
            polar = angles[:, :-1]
            azimuth = angles[:, -1]
        elif angles.shape[1] == 1:
            polar = np.zeros((m, 0))
            azimuth = angles[:, 0]
        else:
            polar = np.zeros((m, 0))
            azimuth = np.zeros(m)

        return r, polar, azimuth

    def _cartesian_to_ternary_spherical(self, matrix: np.ndarray):
        """
        Convert matrix to padded triples, then replace each triple with:
          [ scaling_factor * norm, scaling_factor * angle, scaling_factor * azimuth ]
        Returns:
          matrix_new : (m, n_padded) where for each triple (j,j+1,j+2) we have (norm, angle, az)
          azimuths   : (m, n_triplets) the azimuth columns extracted (useful if you prefer separate)
        """
        matrix = np.asarray(matrix, dtype=float)
        m, n = matrix.shape

        # pad to multiple of 3
        if n % 3 != 0:
            pad = 3 - (n % 3)
            matrix_new = np.column_stack((matrix, np.zeros((m, pad), dtype=float)))
        else:
            matrix_new = matrix.copy()

        _, n_new = matrix_new.shape
        n_triplets = n_new // 3

        azimuths = np.zeros((m, n_triplets))

        # iterate per triplet
        for t in range(n_triplets):
            j = 3 * t
            tri = matrix_new[:, j:j+3]   # shape (m,3)
            r, polar, az = self._cartesian_to_spherical(tri)  # r:(m,1), polar:(m,? ), az:(m,)
            # place scaled values into matrix_new
            matrix_new[:, j] = self.scaling_factor * r[:, 0]
            # polar may be shape (m,0) or (m,1); take first column if exists, else zeros
            if polar.shape[1] >= 1:
                matrix_new[:, j + 1] = self.scaling_factor * polar[:, 0]
            else:
                matrix_new[:, j + 1] = 0.0
            matrix_new[:, j + 2] = self.scaling_factor * az[:]
            azimuths[:, t] = self.scaling_factor * az[:]

        return matrix_new, azimuths

    # Compatibility wrapper: original code used _pairwise_polar and _ternary_spherical names.
    def _pairwise_polar(self, matrix: np.ndarray) -> np.ndarray:
        # pairwise polar function returns packed pairs in same-size matrix
        return self._cartesian_to_pairwise_polar(matrix)

    def _ternary_spherical(self, X: np.ndarray):
        # return ts (norms + angles packed) and az separately, to match earlier expectations
        matrix_new, az = self._cartesian_to_ternary_spherical(X)
        # build ts as concatenation of norm and angle columns: (m, 2 * n_triplets)
        n_cols = matrix_new.shape[1]
        n_triplets = n_cols // 3
        ts_list = []
        for t in range(n_triplets):
            j = 3 * t
            ts_list.append(matrix_new[:, j])     # norm
            ts_list.append(matrix_new[:, j + 1]) # angle
        if len(ts_list) > 0:
            ts = np.column_stack(ts_list)
        else:
            ts = np.zeros((matrix_new.shape[0], 0))
        return ts, az

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
        Compute Gaussian PDFs in a fully log-safe way on input X (which is expected to match self.means_/covs_ space).
        Returns normal-space pdf values (stabilized).
        """
        X = np.asarray(X)
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

        # stabilized back to normal-space pdfs
        max_log = np.max(logpdfs, axis=1, keepdims=True)
        pdfs = np.exp(logpdfs - max_log)
        denom = (np.sum(pdfs, axis=1, keepdims=True) + 1e-12)
        pdfs = pdfs / denom
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
        X_cart = X.copy()   # original Cartesian data for Gaussian recomputation
        polar = self._pairwise_polar(X)  # EM runs on pairwise-polar space
        
        if y is not None:
            self.classes_ = np.unique(y)
        
        best_ll = -np.inf
        rng = np.random.default_rng(42)
        best = None

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

            if ll > best_ll or best is None:
                best_ll = ll
                best = {
                    "means": self.means_.copy(),
                    "covs": self.covs_.copy(),
                    "weights": self.weights_.copy(),
                    "resp": resp.copy()
                }

        # restore best polar-space mixture params
        self.means_ = best["means"]
        self.covs_ = best["covs"]
        self.weights_ = best["weights"]
        resp = best["resp"]

        # -----------------------
        # Compute ternary spherical packed matrix and azimuths
        # -----------------------
        ts_packed, az = self._cartesian_to_ternary_spherical(X)  # matrix with triples [norm, angle, az]
        # Build ts = concatenation of norm and angle columns (for Gaussian-like magnitude modeling in ts-space)
        n_cols = ts_packed.shape[1]
        n_triplets = n_cols // 3
        ts_list = []
        for t in range(n_triplets):
            j = 3 * t
            ts_list.append(ts_packed[:, j])     # norm
            ts_list.append(ts_packed[:, j + 1]) # angle
        if len(ts_list) > 0:
            ts = np.column_stack(ts_list)
        else:
            ts = np.zeros((ts_packed.shape[0], 0))

        # store ts means/variance (as before)
        self.means_ts_ = np.zeros((self.n_components, ts.shape[1]))
        self.nb_var_ts_ = np.zeros((self.n_components, ts.shape[1]))

        for k in range(self.n_components):
            w = resp[:, k][:, None]
            self.means_ts_[k] = np.sum(w * ts, axis=0) / (np.sum(w) + 1e-12)
            self.nb_var_ts_[k] = np.sum(w * (ts - self.means_ts_[k])**2, axis=0) / (np.sum(w) + 1e-12)
            self.nb_var_ts_[k] += 1e-6

        # -----------------------
        # Compute von Mises params from azimuths (spherical)
        # -----------------------
        X_sin = np.sin(az)
        X_cos = np.cos(az)
        Nk = np.sum(resp, axis=0) + 1e-12
        self.vm_mu_ = np.zeros(self.n_components)
        self.vm_kappa_ = np.zeros(self.n_components)

        for k in range(self.n_components):
            S = np.sum(resp[:, k][:, None] * X_sin, axis=0)
            C = np.sum(resp[:, k][:, None] * X_cos, axis=0)
            S_sum = np.sum(S)
            C_sum = np.sum(C)
            mu = np.arctan2(S_sum, C_sum)
            R_bar = np.sqrt(S_sum*S_sum + C_sum*C_sum) / (Nk[k] * max(1, az.shape[1]))
            self.vm_mu_[k] = mu
            if R_bar < 1e-6:
                kappa = 0.0
            elif R_bar < 0.53:
                kappa = 2*R_bar + R_bar**3 + 5*R_bar**5/6.0
            elif R_bar < 0.85:
                kappa = -0.4 + 1.39*R_bar + 0.43/(1 - R_bar)
            else:
                kappa = 1.0 / (R_bar**3 - 4*R_bar**2 + 3*R_bar)
            self.vm_kappa_[k] = max(kappa, 1e-6)

        # -----------------------
        # NEW: compute Cartesian Gaussian parameters using resp (weighted by polar responsibilities)
        # -----------------------
        n_cart, D_cart = X_cart.shape
        K = self.n_components
        self.means_cart_ = np.zeros((K, D_cart))
        if self.covariance_type == "full":
            self.covs_cart_ = np.zeros((K, D_cart, D_cart))
        else:
            self.covs_cart_ = np.zeros((K, D_cart))

        Nk_cart = np.sum(resp, axis=0) + 1e-12
        for k in range(K):
            w = resp[:, k][:, None]
            Nk_k = Nk_cart[k]
            mu_cart = np.sum(w * X_cart, axis=0) / Nk_k
            self.means_cart_[k] = mu_cart
            diff_cart = X_cart - mu_cart
            if self.covariance_type == "full":
                cov_k = (w * diff_cart).T @ diff_cart / Nk_k
                cov_k += self.reg_covar * np.eye(D_cart)
                self.covs_cart_[k] = cov_k
            else:
                var_k = np.sum(w * (diff_cart**2), axis=0) / Nk_k
                var_k += self.reg_covar
                self.covs_cart_[k] = var_k

        return self

    # helper: Gaussian log-likelihood in Cartesian space
    def _log_gauss_cartesian(self, x_cart, k):
        D = x_cart.shape[0]
        if self.covariance_type == "full":
            cov_k = self.covs_cart_[k] + self.reg_covar * np.eye(D)
            sign, logdet = np.linalg.slogdet(cov_k)
            inv_cov = np.linalg.pinv(cov_k)
            diff = x_cart - self.means_cart_[k]
            mah = float(diff @ inv_cov @ diff)
            log_gauss = -0.5 * (mah + logdet + D * np.log(2 * np.pi))
        elif self.covariance_type == "diag":
            var = self.covs_cart_[k] + self.reg_covar
            diff = x_cart - self.means_cart_[k]
            mah = np.sum((diff**2) / var)
            log_gauss = -0.5 * (mah + np.sum(np.log(2 * np.pi * var)))
        else:  # spherical
            s = float(self.covs_cart_[k] + self.reg_covar)
            diff = x_cart - self.means_cart_[k]
            mah = np.sum(diff**2) / s
            log_gauss = -0.5 * (mah + D * np.log(2 * np.pi * s))
        return log_gauss

    def predict_prob(self, X):
        X = np.asarray(X)
        # get ternary packed matrix + azimuths
        ts_packed, az = self._cartesian_to_ternary_spherical(X)
        # build ts as norms+angles packed (for compatibility)
        n_cols = ts_packed.shape[1]
        n_triplets = n_cols // 3
        ts_list = []
        for t in range(n_triplets):
            j = 3 * t
            ts_list.append(ts_packed[:, j])
            ts_list.append(ts_packed[:, j + 1])
        if len(ts_list) > 0:
            ts = np.column_stack(ts_list)
        else:
            ts = np.zeros((ts_packed.shape[0], 0))

        M = X.shape[0]
        K = self.n_components
        probs = np.zeros((M, K))

        # evaluate Gaussian on Cartesian X, vM on azimuths
        for i in range(M):
            x_cart = X[i]
            xa = az[i]
            logp = []
            for k in range(K):
                # Gaussian log-likelihood in Cartesian using computed Cartesian params
                try:
                    log_gauss = self._log_gauss_cartesian(x_cart, k)
                except Exception:
                    # fallback to ts-based gaussian if cartesian params missing
                    if ts.shape[1] > 0:
                        mean = self.means_ts_[k]
                        var = self.nb_var_ts_[k]
                        diff = ts[i] - mean
                        log_gauss = -0.5 * np.sum(np.log(2*np.pi*var) + (diff*diff)/var)
                    else:
                        log_gauss = 0.0

                # von Mises log-likelihood on azimuths
                kap = float(self.vm_kappa_[k])
                mu = self.vm_mu_[k]
                # sum over az dimensions (if vector)
                try:
                    log_vm = np.sum(kap * np.cos(xa - mu) - np.log(2*np.pi) - np.log(i0(kap) + 1e-12))
                except Exception:
                    log_vm = kap * np.cos(float(xa) - mu) - np.log(2*np.pi) - np.log(i0(kap) + 1e-12)

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
#  SKLEARN-COMPATIBLE WRAPPER (keeps the original API/behavior)
# ===========================================================

class PolarSphericalClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn API wrapper for the Polar-Spherical Transform Mixture Model.
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
        self._y = np.array(y)

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
