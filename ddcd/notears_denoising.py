# Modified based on the original implementation of NOTEARS
# https://github.com/xunzheng/notears/blob/master/notears/linear.py
# Added denoising objective

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid

def linear_beta_schedule(timesteps, start_noise, end_noise):
    scale = 1000 / timesteps
    beta_start = scale * start_noise
    beta_end = scale * end_noise
    return np.linspace(beta_start, beta_end, timesteps, dtype = float)


def notears_linear(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3, start_noise=1e-6):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    Args:
        X (np.ndarray): [n, d] sample matrix
        lambda1 (float): l1 penalty parameter
        loss_type (str): l2, logistic, poisson, denoising
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold

    Returns:
        W_est (np.ndarray): [d, d] estimated DAG
    """
    T = 5000
    start_noise = start_noise
    end_noise = start_noise * 10
    betas = linear_beta_schedule(T, start_noise, end_noise)
    alphas = 1. - betas
    alpha_bars = np.cumprod(alphas, axis=0)
    mean_schedule = np.sqrt(alpha_bars)
    std_schedule = np.sqrt(1. - alpha_bars)
    
    def forward_pass(x):
        t = np.random.randint(0, T, (x.shape[0],))
        noise = np.random.randn(x.shape[0], x.shape[1])
        mean_coef = np.take_along_axis(mean_schedule, t, axis=-1)
        std_coef = np.take_along_axis(std_schedule, t, axis=-1)
        x_t = np.expand_dims(mean_coef, -1) * x + np.expand_dims(std_coef, -1) * noise
        return t, x_t, noise, std_coef

    def _loss(W):
        """Evaluate value and gradient of loss."""
        if loss_type == 'l2':
            M = X @ W
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'denoising':
            t, X_t, noise, std_coef = forward_pass(X)
            Z_pred = X_t - X_t @ W
            Z_exp = (noise - noise @ W) * std_coef.reshape(-1 , 1)
            loss = 0.5 / X.shape[0] * ((Z_pred - Z_exp) ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X_t.T @ (Z_pred - Z_exp)
        elif loss_type == 'logistic':
            M = X @ W
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            M = X @ W
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (Zheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        gradients.append(G_smooth)
        losses.append(obj)
        ws.append(W)
        return obj, g_obj

    n, d = X.shape
    global_g_obj = None
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    global gradients
    global losses
    global ws
    gradients = []
    losses = []
    ws = []
    gradients_output = []
    losses_output = []
    ws_output = []
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)
    for _ in tqdm(range(max_iter)):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
                if len(gradients) != 0:
                    gradients_output.append([x.copy() for x in gradients])
                    losses_output.append(losses.copy())
                    ws_output.append([x.copy() for x in ws])
                    gradients = []
                    losses = []
                    ws = []
            else:
                break
                if len(gradients) != 0:
                    gradients_output.append([x.copy() for x in gradients])
                    losses_output.append(losses.copy())
                    ws_output.append([x.copy() for x in ws])
                    gradients = []
                    losses = []
                    ws = []
        if len(gradients) != 0:
            gradients_output.append([x.copy() for x in gradients])
            losses_output.append(losses.copy())
            ws_output.append([x.copy() for x in ws])
            gradients = []
            losses = []
            ws = []
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est, gradients_output, losses_output, ws_output