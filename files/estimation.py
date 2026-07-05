import numpy as np
import json
import sys
from datetime import datetime
from scipy import stats
from scipy.stats import norm
from scipy import optimize
from scipy.optimize import minimize, minimize_scalar,least_squares
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Настройка русских шрифтов для matplotlib
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False


# ================================================================
# Прогресс бар
# ================================================================

def print_progress(current: int, total: int, prefix: str = '', suffix: str = '',
                   decimals: int = 1, length: int = 40, fill: str = '█'):
    """Отображение прогресс-бара в консоли"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if current == total:
        print()


# ================================================================
# Классы для работы с данными и выводом
# ================================================================

@dataclass
class CensoredData:
    data_name: str
    distr_type: str
    bootstrap: str
    quantiles: List[float]
    data: List[float]
    censored: List[int]

    @classmethod
    def from_dict(cls, data: dict) -> 'CensoredData':
        return cls(
            data_name=data['data_name'],
            distr_type=data['distr_type'],
            bootstrap=data['bootstrap'],
            quantiles=data['quantiles'],
            data=data['data'],
            censored=data.get('censored', [0] * len(data['data']))
        )
#=====================================================================

@dataclass
class BootstrapResult:
    p: float
    quantile_original: float
    quantile_log10: float
    ci_lower: float
    ci_upper: float
    ci_lower_log10: float
    ci_upper_log10: float
    ci_width_log10: float
    bootstrap_log10: np.ndarray
    converged_ratio: float
    n_bootstrap: int
    n_success: int


#=====================================================================

class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def __del__(self):
        self.file.close()


# ================================================================
# Функции распределения Бирнбаума-Сандерса
# ================================================================

def mills_ratio(q: np.ndarray) -> np.ndarray:
    result = np.zeros_like(q)
    mask_small = q <= 20
    mask_large = ~mask_small
    
    if np.any(mask_small):
        q_small = q[mask_small]
        result[mask_small] = norm.pdf(q_small) / np.maximum(1 - norm.cdf(q_small), 1e-10)
    
    if np.any(mask_large):
        q_large = q[mask_large]
        result[mask_large] = q_large + 1.0 / q_large
    
    return result

# ================================================================

def mills_ratio_derivative(q: np.ndarray) -> np.ndarray:
    r_q = mills_ratio(q)
    return r_q * (r_q - q)

# ================================================================

def bs_ppf(p: np.ndarray, alpha: float, beta: float, loc: float) -> np.ndarray:
    z = alpha * norm.ppf(p)
    t = (z + np.sqrt(z*z + 4)) / 2
    return loc + beta * t * t

# ================================================================

def bs_cdf(x: np.ndarray, alpha: float, beta: float, loc: float) -> np.ndarray:
    x_adj = np.maximum(x - loc, 1e-10)
    z = np.sqrt(x_adj / beta) - np.sqrt(beta / x_adj)
    return norm.cdf(z / alpha)


# ================================================================
# Вспомогательные вычисления
# ================================================================

def compute_quantities(x_adj: np.ndarray, alpha: float, beta: float):
    sqrt_x_beta = np.sqrt(x_adj / beta)
    sqrt_beta_x = np.sqrt(beta / x_adj)
    z = sqrt_x_beta - sqrt_beta_x
    w = sqrt_x_beta + sqrt_beta_x
    q = z / alpha
    return z, w, q


# ================================================================
# ПЕРВЫЕ ПРОИЗВОДНЫЕ lnL (аналитические)
# ================================================================

def first_derivatives(alpha: float, beta: float, data: np.ndarray, 
                       censored: np.ndarray, fixed_loc: float) -> Tuple[float, float]:
    x_adj = np.maximum(data - fixed_loc, 1e-10)
    z, w, q = compute_quantities(x_adj, alpha, beta)
    
    failure_mask = censored == 0
    censored_mask = censored == 1
    
    dL_dalpha = 0.0
    if np.any(failure_mask):
        q_fail = q[failure_mask]
        dL_dalpha += np.sum(q_fail * q_fail - 1.0)
    if np.any(censored_mask):
        q_cens = q[censored_mask]
        r_q = mills_ratio(q_cens)
        dL_dalpha += np.sum(q_cens * r_q)
    dL_dalpha /= alpha
    
    dL_dbeta = 0.0
    if np.any(failure_mask):
        z_fail = z[failure_mask]
        w_fail = w[failure_mask]
        term = (z_fail / (2.0 * beta)) * (w_fail / (alpha * alpha) - 1.0 / w_fail)
        dL_dbeta += np.sum(term)
    if np.any(censored_mask):
        w_cens = w[censored_mask]
        q_cens = q[censored_mask]
        r_q = mills_ratio(q_cens)
        dL_dbeta += np.sum(w_cens * r_q / (2.0 * alpha * beta))
    
    return dL_dalpha, dL_dbeta


# ================================================================
# ФУНКЦИЯ НЕВЯЗОК ДЛЯ LEAST_SQUARES
# ================================================================

def residuals_bs(params: np.ndarray, data_scaled: np.ndarray, censored: np.ndarray) -> np.ndarray:
    alpha, beta_scaled = params
    if alpha <= 1e-10 or beta_scaled <= 1e-10:
        return np.array([1e10, 1e10])
    
    z_s = np.sqrt(data_scaled / beta_scaled) - np.sqrt(beta_scaled / data_scaled)
    w_s = np.sqrt(data_scaled / beta_scaled) + np.sqrt(beta_scaled / data_scaled)
    q_s = z_s / alpha
    
    failure_mask = censored == 0
    censored_mask = censored == 1
    
    dL_dalpha = 0.0
    if np.any(failure_mask):
        q_fail = q_s[failure_mask]
        dL_dalpha += np.sum(q_fail * q_fail - 1.0)
    if np.any(censored_mask):
        q_cens = q_s[censored_mask]
        r_q = mills_ratio(q_cens)
        dL_dalpha += np.sum(q_cens * r_q)
    dL_dalpha /= alpha
    
    dL_dbeta = 0.0
    if np.any(failure_mask):
        z_fail = z_s[failure_mask]
        w_fail = w_s[failure_mask]
        term = (z_fail / (2.0 * beta_scaled)) * (w_fail / (alpha * alpha) - 1.0 / w_fail)
        dL_dbeta += np.sum(term)
    if np.any(censored_mask):
        w_cens = w_s[censored_mask]
        q_cens = q_s[censored_mask]
        r_q = mills_ratio(q_cens)
        dL_dbeta += np.sum(w_cens * r_q / (2.0 * alpha * beta_scaled))
    
    return np.array([dL_dalpha, dL_dbeta])


# ================================================================
# ОСНОВНАЯ ФУНКЦИЯ ОЦЕНКИ BS (с фиксированным γ)
# ================================================================

def fit_bs_fixed_loc(data: np.ndarray, censored: np.ndarray, fixed_loc: float) -> dict:
    x_adj = np.maximum(data - fixed_loc, 1e-10)
    failure_data = x_adj[censored == 0]
    
    if len(failure_data) > 0:
        scale_factor = np.exp(np.mean(np.log(failure_data)))
    else:
        scale_factor = np.median(x_adj)
    
    data_scaled = x_adj / scale_factor
    
    failure_scaled = data_scaled[censored == 0]
    if len(failure_scaled) > 0:
        mean_fail = np.mean(failure_scaled)
        var_fail = np.var(failure_scaled)
        cv = np.sqrt(var_fail) / mean_fail if mean_fail > 0 else 0.5
        alpha_init = max(0.1, cv * 1.5)
        beta_scaled_init = mean_fail / (1 + alpha_init * alpha_init / 2)
        alpha_init = np.clip(alpha_init, 0.2, 5.0)
        beta_scaled_init = max(beta_scaled_init, 0.1)
    else:
        alpha_init = 1.0
        beta_scaled_init = 1.0
    
    result = optimize.least_squares(
        residuals_bs,
        [alpha_init, beta_scaled_init],
        method='lm',
        args=(data_scaled, censored),
        max_nfev=500,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12
    )
    
    alpha, beta_scaled = result.x
    alpha = max(0.1, min(50.0, alpha))
    beta_scaled = max(1e-6, beta_scaled)
    beta = beta_scaled * scale_factor
    
    r_final = residuals_bs([alpha, beta_scaled], data_scaled, censored)
    Q_min = np.sum(r_final * r_final)
    
    dL_dalpha, dL_dbeta = first_derivatives(alpha, beta, data, censored, fixed_loc)
    
    median = fixed_loc + beta
    
    return {
        'alpha': alpha,
        'beta': beta,
        'beta_scaled': beta_scaled,
        'scale_factor': scale_factor,
        'loc': fixed_loc,
        'Q_min': Q_min,
        'dL_dalpha': dL_dalpha,
        'dL_dbeta': dL_dbeta,
        'success': result.success,
        'message': result.message,
        'data_original': data,
        'data_scaled': data_scaled,
        'censored': censored,
        'median': median,
        'nfev': result.nfev
    }


# ================================================================
# Оценка параметров BS с фиксированным γ (численная)
# ================================================================

def fit_bs_fixed_loc_num(data: np.ndarray, censored: np.ndarray, fixed_loc: float) -> dict:
    """
    Оценка параметров BS распределения с фиксированным порогом γ
    с использованием численных методов (аналогично IG)
    """
    x_adj = np.maximum(data - fixed_loc, 1e-10)
    failure_data = x_adj[censored == 0]
    
    # Начальные приближения
    if len(failure_data) > 0:
        mean_fail = np.mean(failure_data)
        var_fail = np.var(failure_data)
        cv = np.sqrt(var_fail) / mean_fail if mean_fail > 0 else 0.5
        alpha_init = max(0.1, cv * 1.5)
        beta_init = mean_fail / (1 + alpha_init * alpha_init / 2)
        alpha_init = np.clip(alpha_init, 0.2, 5.0)
        beta_init = max(beta_init, 0.1)
    else:
        alpha_init = 1.0
        beta_init = np.median(x_adj)
    
    # Оптимизация
    result = minimize(
        neg_log_likelihood_bs,
        [alpha_init, beta_init],
        args=(data, censored),
        method='L-BFGS-B',
        bounds=[(1e-6, 50.0), (1e-6, None)],
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
    )
    
    alpha_opt, beta_opt = result.x
    
    # Численный гессиан и ковариационная матрица
    args_fixed = (data, censored)
    hessian = numerical_hessian(neg_log_likelihood_bs, result.x, args_fixed, eps=1e-6)
    
    # Проверка положительной определенности
    eigvals = np.linalg.eigvals(hessian)
    if np.min(eigvals) <= 0:
        # Регуляризация
        lambda_reg = 1e-6 * np.max(np.abs(eigvals))
        hessian = hessian + lambda_reg * np.eye(2)
    
    try:
        cov_matrix = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        cov_matrix = np.linalg.pinv(hessian)
    
    se_alpha = np.sqrt(max(cov_matrix[0, 0], 0))
    se_beta = np.sqrt(max(cov_matrix[1, 1], 0))
    
    # Вычисление градиента в точке оптимума
    grad = numerical_gradient(neg_log_likelihood_bs, result.x, args_fixed, eps=1e-8)
    Q_min = np.sum(grad**2)
    
    # Медиана
    median = fixed_loc + beta_opt * (((alpha_opt * norm.ppf(0.5) + np.sqrt((alpha_opt * norm.ppf(0.5))**2 + 4)) / 2)**2)
    
    return {
        'alpha': alpha_opt,
        'beta': beta_opt,
        'loc': fixed_loc,
        'Q_min': Q_min,
        'grad_final': grad,
        'cov_matrix': cov_matrix,
        'se_alpha': se_alpha,
        'se_beta': se_beta,
        'median': median,
        'success': result.success,
        'message': result.message,
        'data_original': data,
        'censored': censored,
        'nfev': result.nfev,
        'hessian': hessian
    }


# ================================================================
# АВТОМАТИЧЕСКИЙ ПОИСК ОПТИМАЛЬНОГО ПОРОГА γ BS
# ================================================================

def find_optimal_gamma_bs(data:np.ndarray,censored:np.ndarray,min_gamma,max_gamma,tol,max_iter) -> dict:
    
    if max_gamma is None:
        failure_min = np.min(data[censored == 0])
        max_gamma = failure_min - 1e-6
    
    def objective(gamma):
        if gamma < min_gamma or gamma >= max_gamma:
            return 1e10
        try:
            if estimation_method=="analytical": result = fit_bs_fixed_loc(data, censored, gamma)
            if estimation_method=="numerical": result = fit_bs_fixed_loc_num(data, censored, gamma)

            return result['Q_min']
        except Exception:
            return 1e10
    
    res = minimize_scalar(
        objective,
        bounds=(min_gamma, max_gamma),
        method='bounded',
        options={'xatol': tol, 'maxiter': max_iter}
    )
    
    optimal_gamma = res.x
    if estimation_method=="analytical": fit_result = fit_bs_fixed_loc(data, censored,optimal_gamma)
    if estimation_method=="numerical": fit_result = fit_bs_fixed_loc_num(data, censored,optimal_gamma)
    fit_result['optimal_gamma'] = optimal_gamma
    fit_result['gamma_search_converged'] = res.success
    fit_result['gamma_search_iterations'] = res.nfev
    fit_result['gamma_search_message'] = res.message
    
    return fit_result


# ================================================================
# Функция логарифма правдоподобия для BS (численная)
# ================================================================

def neg_log_likelihood_bs(params: np.ndarray, data: np.ndarray, censored: np.ndarray) -> float:
    """
    Отрицательная логарифмическая функция правдоподобия для BS распределения
    params = [alpha, beta, loc] или [alpha, beta] если loc фиксирован
    """
    if len(params) == 3:
        alpha, beta, loc = params
    else:
        alpha, beta = params
        loc = 0.0
    
    # Проверка допустимости параметров
    if alpha <= 1e-10 or beta <= 1e-10:
        return 1e10
    
    # Для распределения с порогом loc должен быть меньше минимальных данных
    if loc >= np.min(data):
        return 1e10
    if loc < 0:
        return 1e10
    
    x_adj = np.maximum(data - loc, 1e-10)
    if np.any(x_adj <= 0):
        return 1e10
    
    # Вычисление z = sqrt(x/beta) - sqrt(beta/x)
    z = np.sqrt(x_adj / beta) - np.sqrt(beta / x_adj)
    q = z / alpha
    
    failure_mask = censored == 0
    censored_mask = censored == 1
    
    nll = 0.0
    
    if np.any(failure_mask):
        # Вклад разрушенных образцов (плотность)
        # log f(x) = -log(alpha) - log(beta) - log(2*pi)/2 + log(z) - 0.5*q^2 - 0.5*log(x_adj/beta)???
        # Стандартная формула: log f(x) = -log(alpha) - 0.5*log(2*pi) - 0.5*q^2 - 0.5*log(x_adj/beta) - log(beta)
        log_pdf = -np.log(alpha) - 0.5 * np.log(2 * np.pi) - 0.5 * q[failure_mask]**2 - 0.5 * np.log(x_adj[failure_mask] / beta) - np.log(beta)
        nll -= np.sum(log_pdf)
    
    if np.any(censored_mask):
        # Вклад цензурированных образцов (функция выживания = 1 - CDF)
        log_sf = np.log(np.maximum(1 - norm.cdf(q[censored_mask]), 1e-10))
        nll -= np.sum(log_sf)
    
    return nll

# ================================================================
# Численные градиенты и гессиан (общие для всех распределений)
# ================================================================

def numerical_gradient(func, params: np.ndarray, args: tuple, eps: float = 1e-8) -> np.ndarray:
    """Численное вычисление градиента методом центральных разностей"""
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        grad[i] = (func(params_plus, *args) - func(params_minus, *args)) / (2 * eps)
    return grad

#========================================================================================

def numerical_hessian(func, params: np.ndarray, args: tuple, eps: float = 1e-6) -> np.ndarray:
    """Численное вычисление гессиана методом центральных разностей"""
    n = len(params)
    hessian = np.zeros((n, n))
    
    # Вычисляем градиенты для каждого параметра
    grad_0 = numerical_gradient(func, params, args, eps)
    
    for i in range(n):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps
        
        grad_plus = numerical_gradient(func, params_plus, args, eps)
        grad_minus = numerical_gradient(func, params_minus, args, eps)
        
        hessian[:, i] = (grad_plus - grad_minus) / (2 * eps)
    
    # Симметризуем гессиан
    hessian = (hessian + hessian.T) / 2
    
    return hessian

# ================================================================
# Функции для обратного нормального распределения (IG)
# ================================================================

def invgauss_cdf(x: np.ndarray, mu: float, lam: float, loc: float) -> np.ndarray:
    x_adj = x - loc 
    sqrt_lam_x = np.sqrt(lam / x_adj)
    term1 = sqrt_lam_x * (x_adj / mu - 1)
    term2 = sqrt_lam_x * (x_adj / mu + 1)
    
    cdf = stats.norm.cdf(term1) + np.exp(2 * lam / mu) * stats.norm.cdf(-term2)
    cdf = np.clip(cdf, 1e-10, 1 - 1e-10)
    return cdf

# ================================================================

def invgauss_pdf(x: np.ndarray, mu: float, lam: float, loc: float) -> np.ndarray:
    """Плотность IG с порогом loc (γ)"""
    x_adj = x - loc
    pdf = np.sqrt(lam / (2 * np.pi * x_adj**3)) * np.exp(-lam * (x_adj - mu)**2 / (2 * mu**2 * x_adj))
    return pdf

# ================================================================

def invgauss_ppf(p: float, mu: float, lam: float, loc: float) -> float:
    if p <= 0:
        return loc
    if p >= 1:
        return np.inf
    q = stats.invgauss.ppf(p, mu=mu/lam, scale=lam)
    return loc + q

# ================================================================

def invgauss_ppf_1(p: np.ndarray, mu: float, lam: float, loc: float) -> np.ndarray:
    p = np.asarray(p)
    original_shape = p.shape
    p_flat = p.flatten()
    
    result = np.zeros_like(p_flat)
    
    for idx, p_val in enumerate(p_flat):
        if p_val <= 0:
            result[idx] = loc
        elif p_val >= 1:
            result[idx] = np.inf
        else:
            q_scaled = stats.invgauss.ppf(p_val, mu=mu/lam, scale=lam)
            result[idx] = loc + q_scaled
    
    return result.reshape(original_shape)

# ================================================================

def invgauss_logpdf(x: np.ndarray, mu: float, lam: float, loc: float) -> np.ndarray:
    """Логарифм плотности IG с порогом loc (γ)"""
    x_adj = np.maximum(x - loc, 1e-10)
    log_pdf = 0.5 * (np.log(lam) - np.log(2 * np.pi) - 3 * np.log(x_adj))
    log_pdf -= lam * (x_adj - mu) ** 2 / (2 * mu ** 2 * x_adj)
    return log_pdf

# ================================================================

def invgauss_logsf(x: np.ndarray, mu: float, lam: float, loc: float) -> np.ndarray:
    """Логарифм функции выживания IG"""
    cdf = invgauss_cdf(x, mu, lam, loc)
    return np.log(np.maximum(1 - cdf, 1e-10))

# ================================================================

def invgauss_cdf_scaled(x_adj: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """Функция распределения для масштабированных данных (loc=0)"""
    x_adj = np.maximum(x_adj, 1e-10)
    sqrt_lam_x = np.sqrt(lam / x_adj)
    term1 = sqrt_lam_x * (x_adj / mu - 1)
    term2 = sqrt_lam_x * (x_adj / mu + 1)
    
    cdf = stats.norm.cdf(term1) + np.exp(2 * lam / mu) * stats.norm.cdf(-term2)
    cdf = np.clip(cdf, 1e-10, 1 - 1e-10)
    return cdf


# ================================================================
# Аналитические производные логарифма правдоподобия
# ================================================================

def invgauss_logpdf_grad(x_adj: np.ndarray, mu: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """Производные логарифма плотности по μ и λ (для разрушенных образцов)"""
    x_adj = np.maximum(x_adj, 1e-10)
    dlogpdf_dmu = lam * (x_adj - mu) / (mu ** 3)
    dlogpdf_dlam = 0.5 / lam - (x_adj - mu) ** 2 / (2 * mu ** 2 * x_adj)
    return dlogpdf_dmu, dlogpdf_dlam

# ================================================================

def invgauss_logsf_grad(x_adj: np.ndarray, mu: float, lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """Производные логарифма функции выживания по μ и λ (для цензурированных образцов)"""
    x_adj = np.maximum(x_adj, 1e-10)
    sqrt_lam_x = np.sqrt(lam / x_adj)
    term1 = sqrt_lam_x * (x_adj / mu - 1)
    term2 = sqrt_lam_x * (x_adj / mu + 1)
    
    cdf = invgauss_cdf_scaled(x_adj, mu, lam)
    S = 1 - cdf
    S = np.maximum(S, 1e-10)
    
    phi_term1 = stats.norm.pdf(term1)
    phi_term2 = stats.norm.pdf(term2)
    
    dterm1_dmu = -sqrt_lam_x * x_adj / (mu ** 2)
    dterm2_dmu = -sqrt_lam_x * x_adj / (mu ** 2)
    
    dterm1_dlam = 0.5 * sqrt_lam_x / lam * (x_adj / mu - 1)
    dterm2_dlam = 0.5 * sqrt_lam_x / lam * (x_adj / mu + 1)
    
    exp_term = np.exp(2 * lam / mu)
    dexp_dmu = -2 * lam * exp_term / (mu ** 2)
    dexp_dlam = 2 * exp_term / mu
    
    dF_dmu = phi_term1 * dterm1_dmu + dexp_dmu * stats.norm.cdf(-term2) + exp_term * phi_term2 * (-dterm2_dmu)
    dF_dlam = phi_term1 * dterm1_dlam + dexp_dlam * stats.norm.cdf(-term2) + exp_term * phi_term2 * (-dterm2_dlam)
    
    dlogS_dmu = -dF_dmu / S
    dlogS_dlam = -dF_dlam / S
    
    return dlogS_dmu, dlogS_dlam


# ================================================================
# Функция правдоподобия и её градиент
# ================================================================

def neg_log_likelihood_grad(params_scaled: np.ndarray, x_adj: np.ndarray, censored: np.ndarray,
                            mu_scale: float, lam_scale: float) -> np.ndarray:
    """
    Градиент отрицательной логарифмической функции правдоподобия.
    params_scaled = [mu/mu_scale, lam/lam_scale] — масштабированные параметры
    """
    mu_scaled, lam_scaled = params_scaled
    mu = mu_scaled * mu_scale
    lam = lam_scaled * lam_scale
    
    if mu <= 0 or lam <= 0:
        return np.array([1e10, 1e10])
    
    failure_mask = censored == 0
    censored_mask = censored == 1
    
    grad_mu = 0.0
    grad_lam = 0.0
    
    if np.any(failure_mask):
        x_fail = x_adj[failure_mask]
        dlogpdf_dmu, dlogpdf_dlam = invgauss_logpdf_grad(x_fail, mu, lam)
        grad_mu -= np.sum(dlogpdf_dmu)
        grad_lam -= np.sum(dlogpdf_dlam)
    
    if np.any(censored_mask):
        x_cens = x_adj[censored_mask]
        dlogS_dmu, dlogS_dlam = invgauss_logsf_grad(x_cens, mu, lam)
        grad_mu -= np.sum(dlogS_dmu)
        grad_lam -= np.sum(dlogS_dlam)
    
    # Возвращаем градиент ПО МАСШТАБИРОВАННЫМ параметрам
    return np.array([grad_mu * mu_scale, grad_lam * lam_scale])

# ================================================================

def residuals_ig(params_scaled: np.ndarray, x_adj: np.ndarray, censored: np.ndarray,
                 mu_scale: float, lam_scale: float) -> np.ndarray:
    """Невязки = градиент"""
    mu_s, lam_s = params_scaled
    if mu_s <= 0 or lam_s <= 0:  return np.array([1e10, 1e10])
    return neg_log_likelihood_grad(params_scaled, x_adj, censored, mu_scale, lam_scale)

# ================================================================
# Функция логарифма правдоподобия для IG (для оптимизации)
# ================================================================

def neg_log_likelihood_ig(params: np.ndarray, data: np.ndarray, censored: np.ndarray) -> float:
    """
    Отрицательная логарифмическая функция правдоподобия для IG
    params = [mu, lam, loc] или [mu, lam] если loc фиксирован
    """
    if len(params) == 3:
        mu, lam, loc = params
    else:
        mu, lam = params
        loc = 0.0
    
    # Проверка допустимости параметров
    if mu <= 0 or lam <= 0:
        return 1e10
    
    # Для неусеченного распределения loc должен быть меньше минимальных данных
    if loc >= np.min(data):
        return 1e10
    if loc < 0:
        return 1e10
    
    x_adj = data - loc
    if np.any(x_adj <= 0):
        return 1e10
    
    failure_mask = censored == 0
    censored_mask = censored == 1
    
    nll = 0.0
    
    if np.any(failure_mask):
        # Вклад разрушенных образцов (плотность)
        nll -= np.sum(invgauss_logpdf(data[failure_mask], mu, lam, loc))
    
    if np.any(censored_mask):
        # Вклад цензурированных образцов (функция выживания)
        nll -= np.sum(invgauss_logsf(data[censored_mask], mu, lam, loc))
    
    return nll


# ================================================================
# Оценка параметров IG с фиксированным γ
# ================================================================

def fit_ig_fixed_loc(data: np.ndarray, censored: np.ndarray, fixed_loc: float) -> dict:
    """
    Оценка параметров IG-распределения с фиксированным порогом γ.
    """
    # Вычитаем порог
    x_adj = np.maximum(data - fixed_loc, 1e-10)
    
    # Масштабирование данных для численной устойчивости
    failure_data = x_adj[censored == 0]
    #scale_factor=1.0
    if len(failure_data) > 0:
        scale_factor = np.exp(np.mean(np.log(failure_data)))
    else:
        scale_factor = np.median(x_adj)
    
    x_scaled = x_adj / scale_factor
    
    # Начальные приближения (масштабированные)
    failure_scaled = x_scaled[censored == 0]
    if len(failure_scaled) > 0:
        mu_init = np.mean(failure_scaled)
        var_init = np.var(failure_scaled)
        lam_init = mu_init**3 / var_init if var_init > 0 else 1.0
    else:
        mu_init = 1.0
        lam_init = 1.0
    
    mu_init = max(mu_init, 0.1)
    lam_init = max(lam_init, 0.1)
    lam_init = np.clip(lam_init, 0.1, 1e6)
    
    # Масштабы для параметров (чтобы они были порядка 1)
    mu_scale = mu_init
    lam_scale = lam_init
    
    mu_init_scaled = mu_init / mu_scale  # = 1
    lam_init_scaled = lam_init / lam_scale  # = 1
    
    # Оптимизация методом Левенберга-Марквардта
    result = least_squares(
        residuals_ig,
        [mu_init_scaled, lam_init_scaled],
        method='lm',
        args=(x_scaled, censored, mu_scale, lam_scale),
        max_nfev=200,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12
    )
    
    mu_scaled, lam_scaled = result.x
    mu = mu_scaled * mu_scale
    lam = lam_scaled * lam_scale
    mu = max(0.1, mu)
    lam = max(0.1, lam)
    
    # Пересчёт в исходные единицы (умножаем на scale_factor)
    mu_orig = mu * scale_factor
    lam_orig = lam * scale_factor
    
    # Вычисление Q в точке минимума
    grad_final = neg_log_likelihood_grad([mu_scaled, lam_scaled], x_scaled, censored, mu_scale, lam_scale)
    Q_min = grad_final[0]**2 + grad_final[1]**2
    
    # Ковариационная матрица (через численный гессиан, но корректно)
    eps = 1e-6
    hessian = np.zeros((2, 2))
    params0 = np.array([mu_scaled, lam_scaled])
    
    for i in range(2):
        params_plus = params0.copy()
        params_plus[i] += eps
        grad_plus = neg_log_likelihood_grad(params_plus, x_scaled, censored, mu_scale, lam_scale)
        params_minus = params0.copy()
        params_minus[i] -= eps
        grad_minus = neg_log_likelihood_grad(params_minus, x_scaled, censored, mu_scale, lam_scale)
        hessian[:, i] = (grad_plus - grad_minus) / (2 * eps)
    
    lambda_reg = 1e-8 * np.trace(np.abs(hessian))
    hessian_reg = hessian + lambda_reg * np.eye(2)
    
    try:
        cov_scaled = np.linalg.inv(hessian_reg)
    except np.linalg.LinAlgError:
        cov_scaled = np.linalg.pinv(hessian_reg)
    
    # Ковариационная матрица в исходных единицах (с учётом масштаба параметров и данных)
    J = np.array([[mu_scale, 0], [0, lam_scale]])  # якобиан преобразования параметров
    cov_params = J @ cov_scaled @ J.T  # ковариация (μ, λ) в масштабе x_scaled
    
    # Умножаем на scale_factor для исходных единиц
    cov_matrix = cov_params * (scale_factor ** 2)
    
    se_mu = np.sqrt(max(cov_matrix[0, 0], 0))
    se_lam = np.sqrt(max(cov_matrix[1, 1], 0))
    
    # Медиана: γ + IG_0.5(μ, λ)
    median = fixed_loc + invgauss_ppf(np.array([0.5]), mu_orig, lam_orig, 0)[0]
    
    # Среднее и дисперсия
    mean_val = fixed_loc + mu_orig
    var_val = mu_orig**3 / lam_orig
    std_val = np.sqrt(var_val)
    
    return {
        'mu': mu_orig,
        'lam': lam_orig,
        'mu_scaled': mu,
        'lam_scaled': lam,
        'mu_scale': mu_scale,
        'lam_scale': lam_scale,
        'scale_factor': scale_factor,
        'loc': fixed_loc,
        'Q_min': Q_min,
        'grad_final_mu': grad_final[0],
        'grad_final_lam': grad_final[1],
        'cov_matrix': cov_matrix,
        'se_mu': se_mu,
        'se_lam': se_lam,
        'median': median,
        'mean': mean_val,
        'std': std_val,
        'success': result.success,
        'message': result.message,
        'data_original': data,
        'data_scaled': x_scaled,
        'censored': censored,
        'nfev': result.nfev
    }


# ================================================================
# Оценка параметров IG с фиксированным γ
# ================================================================

def fit_ig_fixed_loc_num(data: np.ndarray, censored: np.ndarray, fixed_loc: float) -> dict:
    
    # Начальные приближения
    failure_data = data[censored == 0]
    failure_data_adj = failure_data - fixed_loc
    
    if len(failure_data) > 0:
        mu_init = np.mean(failure_data_adj)
        var_init = np.var(failure_data_adj)
        lam_init = mu_init**3 / var_init if var_init > 0 else 1.0
    else:
        mu_init = np.median(data - fixed_loc)
        lam_init = 1.0
    
    mu_init = max(mu_init, 0.1)
    lam_init = max(lam_init, 0.1)
    
    # Оптимизация
    result = minimize(
        neg_log_likelihood_ig,
        [mu_init, lam_init],
        args=(data, censored),
        method='L-BFGS-B',
        bounds=[(1e-6, None), (1e-6, None)],
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
    )
    
    mu_opt, lam_opt = result.x
    
    # Численный гессиан и ковариационная матрица
    args_fixed = (data, censored)
    hessian = numerical_hessian(neg_log_likelihood_ig, result.x, args_fixed, eps=1e-6)
    
    # Проверка положительной определенности
    eigvals = np.linalg.eigvals(hessian)
    if np.min(eigvals) <= 0:
        # Регуляризация
        lambda_reg = 1e-6 * np.max(np.abs(eigvals))
        hessian = hessian + lambda_reg * np.eye(2)
    
    try:
        cov_matrix = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        cov_matrix = np.linalg.pinv(hessian)
    
    se_mu = np.sqrt(max(cov_matrix[0, 0], 0))
    se_lam = np.sqrt(max(cov_matrix[1, 1], 0))
    
    # Вычисление градиента в точке оптимума
    grad = numerical_gradient(neg_log_likelihood_ig, result.x, args_fixed, eps=1e-8)
    Q_min = np.sum(grad**2)
    
    # Медиана, среднее, дисперсия
    median = fixed_loc + stats.invgauss.ppf(0.5, mu=mu_opt/lam_opt, scale=lam_opt)
    mean_val = fixed_loc + mu_opt
    var_val = mu_opt**3 / lam_opt
    std_val = np.sqrt(var_val)
    
    return {
        'mu': mu_opt,
        'lam': lam_opt,
        'loc': fixed_loc,
        'Q_min': Q_min,
        'grad_final': grad,
        'cov_matrix': cov_matrix,
        'se_mu': se_mu,
        'se_lam': se_lam,
        'median': median,
        'mean': mean_val,
        'std': std_val,
        'success': result.success,
        'message': result.message,
        'data_original': data,
        'censored': censored,
        'nfev': result.nfev,
        'hessian': hessian
    }


# ================================================================
# АВТОМАТИЧЕСКИЙ ПОИСК ОПТИМАЛЬНОГО ПОРОГА γ IG
# ================================================================

def find_optimal_gamma_ig(data, censored, min_gamma_ratio, max_gamma_ratio, tol, max_iter):

    failure_min = np.min(data[censored == 0])
    min_gamma = failure_min * min_gamma_ratio
    max_gamma = failure_min * max_gamma_ratio
    
    print(f"  Диапазон поиска γ: [{min_gamma:.2f}, {max_gamma:.2f}]")
    
    # Получаем эмпирическую CDF один раз
    km_times, km_survival, km_cdf, km_var, km_se = kaplan_meier(data, censored)
    
    def objective(gamma):
        try:
            # Оцениваем параметры IG при фиксированном γ
            if estimation_method=="analytical": res=fit_ig_fixed_loc(data,censored,gamma)
            if estimation_method=="numerical": res=fit_ig_fixed_loc_num(data,censored,gamma)
            mu, lam, loc = res['mu'], res['lam'], res['loc']
            # Теоретическая CDF в точках КМ
            theo_cdf = invgauss_cdf(km_times, mu, lam, loc)
            # MSE
            mse = np.mean((km_cdf - theo_cdf)**2)
            return mse
        except:
            return 1e10
    
    result = minimize_scalar(objective, bounds=(min_gamma, max_gamma), 
                             method='bounded', options={'xatol': tol, 'maxiter': max_iter})
    
    optimal_gamma = result.x
    if estimation_method=="analytical": fit_result=fit_ig_fixed_loc(data,censored,optimal_gamma)
    if estimation_method=="numerical": fit_result=fit_ig_fixed_loc_num(data,censored,optimal_gamma)

    fit_result['optimal_gamma'] = optimal_gamma
    fit_result['gamma_search_converged'] = result.success
    fit_result['gamma_search_iterations'] = result.nfev
    fit_result['gamma_search_message'] = result.message
    
    return fit_result


# ================================================================
# Функции распределения Вейбулла
# ================================================================

def weibull_cdf(x: np.ndarray, shape: float, scale: float, loc: float) -> np.ndarray:
    x_adj = np.maximum(x - loc, 1e-10)
    return 1.0 - np.exp(-((x_adj / scale) ** shape))


def weibull_ppf(p: np.ndarray, shape: float, scale: float, loc: float) -> np.ndarray:
    return loc + scale * (-np.log(1 - np.array(p))) ** (1.0 / shape)


# ================================================================
# Минимизируемая функция Q = (c*c) для Вейбулла
# ================================================================

def weibull_min_function(b: float, data: np.ndarray, censored: np.ndarray, fixed_loc: float) -> float:
    if b <= 0:
        return 1e10
    x_adj = np.maximum(data - fixed_loc, 1e-10)
    k = np.sum(censored == 0)
    if k == 0:
        return 1e10
    s1 = np.sum(x_adj ** b)
    c = s1 / k

    s2 = 0.0
    s3 = 0.0
    for i in range(len(data)):
        z = (x_adj[i] ** b) / c
        s3 += z * np.log(z)
        if censored[i] == 0:
            s2 += np.log(z)

    c_val = s3 - s2 - k
    return c_val * c_val


# ================================================================
# Оценка параметров с фиксированным γ для Вейбулла
# ================================================================

def fit_weibull_fixed_loc(data: np.ndarray, censored: np.ndarray, fixed_loc: float) -> dict:
    from scipy import optimize
    from scipy.special import gamma

    x_adj = np.maximum(data - fixed_loc, 1e-10)
    failure_data = x_adj[censored == 0]

    if len(failure_data) > 1:
        log_failure = np.log(failure_data)
        std_log = np.std(log_failure)
        shape_init = max(0.5, 1.2 / std_log if std_log > 0 else 1.0)
    else:
        shape_init = 1.0

    shape_init = np.clip(shape_init, 0.2, 10.0)

    result = optimize.minimize_scalar(
        lambda b: weibull_min_function(b, data, censored, fixed_loc),
        bounds=(0.1, 50.0),
        method='bounded'
    )

    shape = result.x
    shape = max(0.1, min(50.0, shape))

    k = np.sum(censored == 0)
    s1 = np.sum(x_adj ** shape)
    scale = (s1 / k) ** (1.0 / shape) if s1 > 0 and shape > 0 else 1.0

    Q_min = weibull_min_function(shape, data, censored, fixed_loc)

    # Ковариационная матрица
    aw = np.log(scale)
    sw = 1.0 / shape
    n = len(data)
    z = (np.log(x_adj) - aw) / sw
    z = np.clip(z, -50, 50)

    s1_cov = 0.0
    s2_cov = 0.0
    for i in range(n):
        if censored[i] == 0:
            s1_cov += z[i]
        s2_cov += z[i] * z[i] * np.exp(z[i])

    k_fail = np.sum(censored == 0)
    info = np.array([[k_fail / n, (k_fail + s1_cov) / n], [(k_fail + s1_cov) / n, (k_fail + s2_cov) / n]])

    lambda_reg = 1e-6 * np.trace(np.abs(info)) / 2
    info = info + lambda_reg * np.eye(2)

    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(info)

    cov_matrix = np.array([
        [cov[0, 0], cov[0, 1] / scale],
        [cov[1, 0] / scale, cov[1, 1] / (scale * scale)]
    ])

    se_shape = np.sqrt(max(cov_matrix[0, 0], 0))
    se_scale = np.sqrt(max(cov_matrix[1, 1], 0))

    median = fixed_loc + scale * (-np.log(0.5)) ** (1.0 / shape)
    mean_val = fixed_loc + scale * gamma(1 + 1.0 / shape)
    std_val = scale * np.sqrt(gamma(1 + 2.0 / shape) - (gamma(1 + 1.0 / shape)) ** 2)

    return {
        'shape': shape,
        'scale': scale,
        'loc': fixed_loc,
        'Q': Q_min,
        'cov_matrix': cov_matrix,
        'se_shape': se_shape,
        'se_scale': se_scale,
        'median': median,
        'mean': mean_val,
        'std': std_val,
        'success': result.success,
        'message': result.message,
        'data_original': data,
        'censored': censored,
        'nfev': result.nfev
    }


# ================================================================
# АВТОМАТИЧЕСКИЙ ПОИСК ОПТИМАЛЬНОГО γ ДЛЯ ВЕЙБУЛЛА
# ================================================================

def find_optimal_gamma_w(data: np.ndarray, censored: np.ndarray,
                       min_gamma_ratio: float, max_gamma_ratio: float,
                       tol: float = 1e-6, max_iter: int = 50) -> dict:
    from scipy.optimize import minimize_scalar

    failure_min = np.min(data[censored == 0])
    min_gamma = failure_min * min_gamma_ratio
    max_gamma = failure_min * max_gamma_ratio

    def objective(gamma):
        if gamma < min_gamma or gamma > max_gamma:
            return 1e10
        try:
            res = fit_weibull_fixed_loc(data, censored, gamma)
            return res['Q']
        except Exception:
            return 1e10

    result = minimize_scalar(objective, bounds=(min_gamma, max_gamma),
                             method='bounded', options={'xatol': tol, 'maxiter': max_iter})

    optimal_gamma = result.x
    fit_result = fit_weibull_fixed_loc(data, censored, optimal_gamma)
    fit_result['optimal_gamma'] = optimal_gamma
    fit_result['gamma_search_converged'] = result.success
    fit_result['gamma_search_iterations'] = result.nfev
    fit_result['gamma_search_message'] = result.message

    return fit_result


# ================================================================
# Оценка Каплана-Мейера
# ================================================================

def kaplan_meier(data: np.ndarray, censored: np.ndarray):
    n = len(data)
    idx = np.argsort(data)
    sorted_data = data[idx]
    sorted_censored = censored[idx]

    times = []
    survival = []
    var = []

    S = 1.0
    V = 0.0

    for i in range(n):
        if sorted_censored[i] == 0:
            at_risk = n - i
            if at_risk > 1:
                S = S * (at_risk - 1) / at_risk
                V = V + 1 / (at_risk * (at_risk - 1))
            else:
                S = 0.0
            times.append(sorted_data[i])
            survival.append(S)
            var.append(S * S * V if S > 0 else 0)

    survival = np.array(survival)
    cdf = 1 - survival
    var = np.array(var)
    se = np.sqrt(var)

    return np.array(times), survival, cdf, var, se

# ================================================================

def kaplan_meier_quantile(km_times: np.ndarray, km_cdf: np.ndarray, p: float) -> float:
    if p <= 0:
        return km_times[0]
    if p >= 1:
        return km_times[-1]

    for i in range(len(km_cdf)):
        if km_cdf[i] >= p:
            if i == 0:
                return km_times[0]
            else:
                t1, t2 = km_times[i-1], km_times[i]
                c1, c2 = km_cdf[i-1], km_cdf[i]
                if c2 > c1:
                    return t1 + (t2 - t1) * (p - c1) / (c2 - c1)
                else:
                    return t1
    return km_times[-1]


# ================================================================
# Дельта-метод для квантили КМ
# ================================================================

def delta_method_km_quantile_ci(km_times: np.ndarray, km_survival: np.ndarray,
                                 km_var: np.ndarray, p: float, conf_level: float = 0.95) -> dict:
    from scipy.stats import norm

    alpha = 1 - conf_level
    z_alpha = norm.ppf(1 - alpha/2)

    q = kaplan_meier_quantile(km_times, 1 - km_survival, p)

    idx = np.argmin(np.abs(km_times - q))

    if idx > 0 and idx < len(km_times) - 1:
        dt = km_times[idx+1] - km_times[idx-1]
        df = (km_survival[idx+1] - km_survival[idx-1]) / dt if dt > 0 else 1
        if df > 0 and km_var[idx] > 0:
            se_q = np.sqrt(km_var[idx]) / abs(df)
        else:
            se_q = np.sqrt(km_var[idx]) * q if km_var[idx] > 0 else q * 0.1
    else:
        se_q = np.sqrt(km_var[idx]) * q if km_var[idx] > 0 else q * 0.1

    ci_lower = q - z_alpha * se_q
    ci_upper = q + z_alpha * se_q

    return {
        'quantile': q,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_lower_log10': np.log10(ci_lower),
        'ci_upper_log10': np.log10(ci_upper),
        'width_log10': np.log10(ci_upper) - np.log10(ci_lower)
    }


# ================================================================
# ПАРАМЕТРИЧЕСКИЙ БУТСТРЕП ДЛЯ BS
# ================================================================

def parametric_bootstrap_bs(
    data: np.ndarray,
    censored: np.ndarray,
    alpha_hat: float,
    beta_hat: float,
    gamma: float,
    p: float,
    n_bootstrap: int = 500,
    conf_level: float = 0.95,
    random_seed: int = 42,
    alpha_min: float = 0.2,
    alpha_max: float = 10.0,
    beta_min_ratio: float = 0.01,
    beta_max_ratio: float = 100.0
) -> BootstrapResult:

    np.random.seed(random_seed)
    n = len(data)
    alpha = 1 - conf_level
    
    data_corrected = data - gamma
    
    # Исходная оценка квантиля (в исходной шкале)
    q_adj_original = bs_ppf(np.array([p]), alpha_hat, beta_hat, 0.0)[0]
    q_original = q_adj_original + gamma
    log10_original = np.log10(q_original)
    
    bootstrap_log10 = []
    n_success = 0
    n_rejected = 0
    
    print(f"\n  Параметрический бутстреп для p={p}:")
    print(f"    Исходная квантиль = {q_original:.2e} циклов, log10 = {log10_original:.6f}")
    print(f"    Границы отбраковки: α ∈ [{alpha_min}, {alpha_max}], β ∈ [{beta_min_ratio*beta_hat:.2e}, {beta_max_ratio*beta_hat:.2e}]")
    
    for b in range(n_bootstrap):
        # Генерация выборки (в скорректированной шкале, loc=0)
        u = np.random.uniform(0, 1, n)
        t_adj_full = bs_ppf(u, alpha_hat, beta_hat, 0.0)
        
        # Наложение цензуры
        t_adj_boot = t_adj_full.copy()
        for i in range(n):
            if censored[i] == 1 and t_adj_full[i] > data_corrected[i]:
                t_adj_boot[i] = data_corrected[i]
        
        try:
            if estimation_method=="analytical": result = fit_bs_fixed_loc(t_adj_boot,censored,0)
            if estimation_method=="numerical": result = fit_bs_fixed_loc_num(t_adj_boot,censored,0)
            alpha_b = result['alpha']
            beta_b = result['beta']
            
            # ОТБРАКОВКА ВЫБРОСОВ
            is_valid = (alpha_min < alpha_b < alpha_max and 
                        beta_min_ratio * beta_hat < beta_b < beta_max_ratio * beta_hat)
            
            if is_valid:
                q_adj_b = bs_ppf(np.array([p]), alpha_b, beta_b, 0.0)[0]
                q_b = q_adj_b + gamma
                
                if q_b > 0 and not np.isnan(q_b) and not np.isinf(q_b):
                    log10_b = np.log10(q_b)
                    if log10_b < 15:
                        bootstrap_log10.append(log10_b)
                        n_success += 1
                    else:
                        n_rejected += 1
                else:
                    n_rejected += 1
            else:
                n_rejected += 1
        except Exception:
            n_rejected += 1
    
        # Прогресс-бар
        if (b + 1) % 50 == 0 or b == n_bootstrap - 1:
            print_progress(b + 1, n_bootstrap,
                           prefix=f'    Итерация',
                           suffix=f'успешно: {n_success}, отклонено: {n_rejected}', 
                           length=30)

    converged_ratio = n_success / n_bootstrap
    
    print(f"\n    Успешных: {n_success}/{n_bootstrap} ({converged_ratio*100:.1f}%)")
    print(f"    Отклонено: {n_rejected}")
    
    # Доверительные интервалы
    if n_success >= 50:
        ci_lower_log10 = np.percentile(bootstrap_log10, 100 * alpha / 2)
        ci_upper_log10 = np.percentile(bootstrap_log10, 100 * (1 - alpha / 2))
    else:
        print("    ПРЕДУПРЕЖДЕНИЕ: слишком мало успешных итераций!")
        ci_lower_log10 = log10_original - 1.0
        ci_upper_log10 = log10_original + 1.0
    
    ci_lower = 10 ** ci_lower_log10
    ci_upper = 10 ** ci_upper_log10
    ci_width_log10 = ci_upper_log10 - ci_lower_log10
    
    print(f"    95% ДИ log₁₀: [{ci_lower_log10:.6f}, {ci_upper_log10:.6f}]")
    print(f"    95% ДИ циклы: [{ci_lower:.2e}, {ci_upper:.2e}]")
    
    return BootstrapResult(
        p=p,
        quantile_original=q_original,
        quantile_log10=log10_original,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_lower_log10=ci_lower_log10,
        ci_upper_log10=ci_upper_log10,
        ci_width_log10=ci_width_log10,
        bootstrap_log10=np.array(bootstrap_log10),
        converged_ratio=converged_ratio,
        n_bootstrap=n_bootstrap,
        n_success=n_success
    )


# ================================================================
# ПАРАМЕТРИЧЕСКИЙ БУТСТРЕП ДЛЯ IG
# ================================================================

def parametric_bootstrap_ig(
    data: np.ndarray,
    censored: np.ndarray,
    mu_hat: float,
    lam_hat: float,
    gamma: float,
    p: float,
    n_bootstrap: int = 500,
    conf_level: float = 0.95,
    random_seed: int = 42,
    mu_min_ratio: float = 0.01,
    mu_max_ratio: float = 100.0,
    lam_min_ratio: float = 0.01,
    lam_max_ratio: float = 100.0
) -> BootstrapResult:
    """Параметрический бутстреп для IG-распределения"""
    
    np.random.seed(random_seed)
    n = len(data)
    alpha = 1 - conf_level
    
    # Исходная оценка квантиля (в исходной шкале)
    q_adj_original = invgauss_ppf(p, mu_hat, lam_hat, gamma)
    q_original = q_adj_original + gamma
    log10_original = np.log10(q_original)
    
    bootstrap_log10 = []
    n_success = 0
    n_rejected = 0
    
    print(f"\n  Параметрический бутстреп для p={p}:")
    print(f"    Исходная квантиль = {q_original:.2e} циклов, log10 = {log10_original:.6f}")
    print(f"    Границы отбраковки: μ ∈ [{mu_min_ratio*mu_hat:.2e}, {mu_max_ratio*mu_hat:.2e}], "
          f"λ ∈ [{lam_min_ratio*lam_hat:.2e}, {lam_max_ratio*lam_hat:.2e}]")
    
    for b in range(n_bootstrap):
        # Генерация выборки (в скорректированной шкале, loc=0)
        u = np.random.uniform(0, 1, n)
        t_adj_full = invgauss_ppf_1(u, mu_hat, lam_hat, gamma)
        
        # Наложение цензуры (data уже скорректированы)
        t_adj_boot = t_adj_full.copy()
        for i in range(n):
            if censored[i] == 1 and t_adj_full[i] > data[i]:
                t_adj_boot[i] = data[i]
        
        try:
            # Оценка параметров (fixed_loc=0, данные уже скорректированы)
            #result = fit_ig_fixed_loc(t_adj_boot, censored, 0.0)
            if estimation_method=="analytical": result=fit_ig_fixed_loc(t_adj_boot,censored,0.0)
            if estimation_method=="numerical": result=fit_ig_fixed_loc_num(t_adj_boot,censored,0.0)

            mu_b = result['mu']
            lam_b = result['lam']
            
            # Отбраковка выбросов
            is_valid = (mu_min_ratio * mu_hat < mu_b < mu_max_ratio * mu_hat and
                        lam_min_ratio * lam_hat < lam_b < lam_max_ratio * lam_hat)
            
            if is_valid:
                q_adj_b = invgauss_ppf(p, mu_b, lam_b, gamma)
                q_b = q_adj_b + gamma
                
                if q_b > 0 and not np.isnan(q_b) and not np.isinf(q_b):
                    log10_b = np.log10(q_b)
                    if log10_b < 15:
                        bootstrap_log10.append(log10_b)
                        n_success += 1
                    else:
                        n_rejected += 1
                else:
                    n_rejected += 1
            else:
                n_rejected += 1
        except Exception:
            n_rejected += 1
        
        # Прогресс-бар
        if (b + 1) % 50 == 0 or b == n_bootstrap - 1:
            print_progress(b + 1, n_bootstrap,
                           prefix=f'    Итерация',
                           suffix=f'успешно: {n_success}, отклонено: {n_rejected}', 
                           length=30)
    
    converged_ratio = n_success / n_bootstrap
    
    print(f"\n    Успешных: {n_success}/{n_bootstrap} ({converged_ratio*100:.1f}%)")
    print(f"    Отклонено: {n_rejected}")
    
    # Доверительные интервалы
    if n_success >= 50:
        ci_lower_log10 = np.percentile(bootstrap_log10, 100 * alpha / 2)
        ci_upper_log10 = np.percentile(bootstrap_log10, 100 * (1 - alpha / 2))
    else:
        print("    ПРЕДУПРЕЖДЕНИЕ: слишком мало успешных итераций!")
        ci_lower_log10 = log10_original - 1.0
        ci_upper_log10 = log10_original + 1.0
    
    ci_lower = 10 ** ci_lower_log10
    ci_upper = 10 ** ci_upper_log10
    ci_width_log10 = ci_upper_log10 - ci_lower_log10
    
    print(f"    95% ДИ log₁₀: [{ci_lower_log10:.6f}, {ci_upper_log10:.6f}]")
    print(f"    95% ДИ циклы: [{ci_lower:.2e}, {ci_upper:.2e}]")
    
    return BootstrapResult(
        p=p,
        quantile_original=q_original,
        quantile_log10=log10_original,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_lower_log10=ci_lower_log10,
        ci_upper_log10=ci_upper_log10,
        ci_width_log10=ci_width_log10,
        bootstrap_log10=np.array(bootstrap_log10),
        converged_ratio=converged_ratio,
        n_bootstrap=n_bootstrap,
        n_success=n_success
    )


# ================================================================
# ПАРАМЕТРИЧЕСКИЙ БУТСТРЕП ДЛЯ ВЕЙБУЛЛА
# ================================================================

def parametric_bootstrap_w(
    data: np.ndarray,
    censored: np.ndarray,
    shape_hat: float,
    scale_hat: float,
    gamma: float,
    p: float,
    n_bootstrap: int = 500,
    conf_level: float = 0.95,
    random_seed: int = 42
) -> BootstrapResult:
    
    np.random.seed(random_seed)
    n = len(data)
    alpha = 1 - conf_level
    
    # Исходная оценка квантиля (в исходной шкале)
    q_adj_original = weibull_ppf(np.array([p]), shape_hat, scale_hat, 0.0)[0]
    q_original = q_adj_original + gamma
    log10_original = np.log10(q_original)
    
    bootstrap_log10 = []
    n_success = 0
    n_failed = 0
    
    print(f"\n  Параметрический бутстреп для p={p}:")
    print(f"    Исходная квантиль = {q_original:.2e} циклов, log10 = {log10_original:.6f}")
    print(f"    Отбраковка: ОТКЛЮЧЕНА")
    
    for b in range(n_bootstrap):
        # Генерация выборки (в скорректированной шкале, loc=0)
        u = np.random.uniform(0, 1, n)
        t_adj_full = weibull_ppf(u, shape_hat, scale_hat, 0.0)
        
        # Наложение цензуры (data уже скорректированы)
        t_adj_boot = t_adj_full.copy()
        for i in range(n):
            if censored[i] == 1 and t_adj_full[i] > data[i]:
                t_adj_boot[i] = data[i]
        
        try:
            # Оценка параметров (fixed_loc=0, данные уже скорректированы)
            result = fit_weibull_fixed_loc(t_adj_boot, censored, 0.0)
            shape_b = result['shape']
            scale_b = result['scale']
            
            # БЕЗ ОТБРАКОВКИ — принимаем все результаты
            q_adj_b = weibull_ppf(np.array([p]), shape_b, scale_b, 0.0)[0]
            q_b = q_adj_b + gamma
            
            if q_b > 0 and not np.isnan(q_b) and not np.isinf(q_b):
                log10_b = np.log10(q_b)
                if log10_b < 15:
                    bootstrap_log10.append(log10_b)
                    n_success += 1
                else:
                    n_failed += 1
            else:
                n_failed += 1
        except Exception:
            n_failed += 1
        
        # Прогресс-бар
        if (b + 1) % 50 == 0 or b == n_bootstrap - 1:
            print_progress(b + 1, n_bootstrap,
                           prefix=f'    Итерация',
                           suffix=f'успешно: {n_success}, ошибок: {n_failed}', 
                           length=30)
    
    converged_ratio = n_success / n_bootstrap
    
    print(f"\n    Успешных: {n_success}/{n_bootstrap} ({converged_ratio*100:.1f}%)")
    print(f"    Ошибок: {n_failed}")
    
    # Доверительные интервалы
    if n_success >= 50:
        ci_lower_log10 = np.percentile(bootstrap_log10, 100 * alpha / 2)
        ci_upper_log10 = np.percentile(bootstrap_log10, 100 * (1 - alpha / 2))
    else:
        print("    ПРЕДУПРЕЖДЕНИЕ: слишком мало успешных итераций!")
        ci_lower_log10 = log10_original - 1.0
        ci_upper_log10 = log10_original + 1.0
    
    ci_lower = 10 ** ci_lower_log10
    ci_upper = 10 ** ci_upper_log10
    ci_width_log10 = ci_upper_log10 - ci_lower_log10
    
    print(f"    95% ДИ log₁₀: [{ci_lower_log10:.6f}, {ci_upper_log10:.6f}]")
    print(f"    95% ДИ циклы: [{ci_lower:.2e}, {ci_upper:.2e}]")
    
    return BootstrapResult(
        p=p,
        quantile_original=q_original,
        quantile_log10=log10_original,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_lower_log10=ci_lower_log10,
        ci_upper_log10=ci_upper_log10,
        ci_width_log10=ci_width_log10,
        bootstrap_log10=np.array(bootstrap_log10),
        converged_ratio=converged_ratio,
        n_bootstrap=n_bootstrap,
        n_success=n_success
    )


# ================================================================
# Визуализация
# ================================================================

def plot_cdf_kaplan_meier_bs(results: dict):

    fig, ax = plt.subplots(figsize=(14, 8))
    data = results['data_original']
    censored = results['censored']
    alpha = results['alpha']
    beta = results['beta']
    loc = results['loc']
    
    failure_times, failure_survival, failure_cdf, km_var, km_se = kaplan_meier(data, censored)
    failure_times_log = np.log10(failure_times)
    
    x_min = max(loc + 1e-10, np.min(data) * 0.5)
    x_max = np.max(data) * 2
    x_theo = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
    x_theo_log = np.log10(x_theo)
    theo_cdf = bs_cdf(x_theo, alpha, beta, loc)
    
    ax.plot(x_theo_log, theo_cdf, 'b-', linewidth=2.5, label='Теоретическая CDF')
    ax.scatter(failure_times_log, failure_cdf, color='red', s=100, zorder=5,
               edgecolor='black', label='Эмпирическая CDF (Каплан-Мейер)')
    
    ax.set_xlabel('log₁₀(Долговечность), lg(N) (циклы)', fontsize=12)
    ax.set_ylabel('Вероятность разрушения F(N)', fontsize=12)
    ax.set_title(f'Распределение Бирнбаума-Сандерса (оптимальный γ = {loc:.0f})', fontsize=13)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    param_text = (f'α={alpha:.6f}\n'
                  f'β={beta:.2f}\n'
                  f'γ (опт.) = {loc:.0f}\n'
                  f'n = {len(data)} ({np.sum(censored==0)} разруш.)\n'
                  f'Медиана = {results["median"]:.2f} циклов\n')
    
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.tight_layout()
    plt.show()


def plot_cdf_kaplan_meier_ig(results: dict):

    fig, ax = plt.subplots(figsize=(14, 8))
    data = results['data_original']
    censored = results['censored']
    mu = results['mu']
    lam = results['lam']
    loc = results['loc']
    
    failure_times, failure_survival, failure_cdf, km_var, km_se = kaplan_meier(data, censored)
    failure_times_log = np.log10(failure_times)
    
    x_min = max(loc + 1e-10, np.min(data) * 0.5)
    x_max = np.max(data) * 2
    x_theo = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
    x_theo_log = np.log10(x_theo)
    theo_cdf = invgauss_cdf(x_theo, mu, lam, loc)
    
    ax.plot(x_theo_log, theo_cdf, 'b-', linewidth=2.5, label='Теоретическая CDF (IG)')
    ax.scatter(failure_times_log, failure_cdf, color='red', s=100, zorder=5,
               edgecolor='black', label='Эмпирическая CDF (Каплан-Мейер)')
    
    ax.set_xlabel('log₁₀(Долговечность), lg(N) (циклы)', fontsize=12)
    ax.set_ylabel('Вероятность разрушения F(N)', fontsize=12)
    ax.set_title(f'Обратное нормальное распределение (IG)\nоптимальный γ = {loc:.0f}', fontsize=13)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    param_text = (f'μ = {mu:.2e} циклов\n'
                  f'log₁₀(μ) = {np.log10(mu):.4f}\n'
                  f'λ = {lam:.2e}\n'
                  f'log₁₀(λ) = {np.log10(lam):.4f}\n'
                  f'γ (опт.) = {loc:.0f}\n'
                  f'log₁₀(γ) = {np.log10(loc):.2f}\n'
                  f'n = {len(data)} ({np.sum(censored==0)} разруш., {np.sum(censored==1)} ценз.)\n'
                  f'Медиана = {results["median"]:.2e} циклов\n'
                  f'log₁₀(медиана) = {np.log10(results["median"]):.4f}')
    
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def plot_cdf_kaplan_meier_weibull(results: dict):

    data = results['data_original']
    censored = results['censored']
    fig, ax = plt.subplots(figsize=(12, 8))
    km_times, km_survival, km_cdf, km_var, km_se = kaplan_meier(data, censored)
    km_times_log = np.log10(km_times)
    ax.step(km_times_log, km_cdf, where='post', color='blue', linewidth=2, label='CDF (Каплан-Мейер)')
    # Теоретическая CDF Вейбулла
    x_theo = np.logspace(np.log10(results['loc']+1e-10), np.log10(np.max(data) * 2), 1000)
    theo_cdf = weibull_cdf(x_theo, results['shape'], results['scale'], results['loc'])
    ax.plot(np.log10(x_theo), theo_cdf, 'r--', linewidth=2, label='Теоретическая CDF (Вейбулл)')
    ax.set_xlabel('log₁₀(Долговечность), lg(N) (циклы)', fontsize=12)
    ax.set_ylabel('Вероятность разрушения F(N)', fontsize=12)
    ax.set_title('Эмпирическая (Каплан-Мейер) и теоретическая (Вейбулл) функции распределения', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ================================================================

def main():
    
#=======Загрузка данных==================================================

    with open('estimation.json', 'r', encoding='utf-8') as f: params_dict = json.load(f)
    data_params = CensoredData.from_dict(params_dict)
    data_name = data_params.data_name
    distr = data_params.distr_type

    global estimation_method
    estimation_method = params_dict.get('estimation_method', 'analytical')

    boots = data_params.bootstrap
    quantiles = data_params.quantiles
    data = np.array(data_params.data)
    censored = np.array(data_params.censored)

#=============================================================================

    if distr == "BS":  
        output_filename = f"BS_{datetime.now().strftime('%d.%m.%Y_%H.%M')} {estimation_method}.out"
        tee = Tee(output_filename)
        sys.stdout = tee

        print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data Name: {data_name}")
        print(f"Method: {estimation_method}")

        results = find_optimal_gamma_bs(data, censored, 0.0, 0.99,1e-6,50)

        print("=" * 80)
        print("ОЦЕНКА ПАРАМЕТРОВ РАСПРЕДЕЛЕНИЯ БИРНБАУМА-САНДЕРСА")
        print("=" * 80)
        print(f"Количество образцов: {len(data)}")
        print(f"  Разрушено: {np.sum(censored == 0)}")
        print(f"  Цензурировано: {np.sum(censored == 1)}")
        print(f"Диапазон данных: [{np.min(data):.2f}, {np.max(data):.2f}]")
        print(f"Минимальное разрушенное значение: {np.min(data[censored == 0]):.2f}")
        print()
        print("=" * 80)
        print("ОЦЕНКИ ПАРАМЕТРОВ")
        print("=" * 80)
        print(f"\nПОРОГ γ = {results['optimal_gamma']:.2f} циклов")
        print(f"  Поиск сошёлся: {results['gamma_search_converged']}")
        print(f"  Число итераций: {results['gamma_search_iterations']}")
        print(f"  Сообщение: {results['gamma_search_message']}")
        print()
        print(f"  alpha (α) = {results['alpha']:.8f}")
        print(f"  beta (β)  = {results['beta']:.2f}")
        print(f"  Медиана   = {results['median']:.2f} циклов")
        mean_val = results['loc'] + results['beta'] * (1 + results['alpha']**2 / 2)
        print(f"  Среднее   = {mean_val:.2f} циклов")
        var_val = (results['beta']**2) * (results['alpha']**2) * (1 + 5 * results['alpha']**2 / 4)
        print(f"  Ст.откл.  = {np.sqrt(var_val):.2f} циклов")
        print(f"\nМИНИМИЗИРУЕМАЯ ФУНКЦИЯ:")
        print(f"  Q = {results['Q_min']:.12e}")
        print(f"\nСТАТИСТИКА ОПТИМИЗАЦИИ:")
        print(f"  Число оценок функции: {results['nfev']}")
        print(f"  Успех: {results['success']}")
        print(f"  Сообщение: {results['message']}")

#=============================================================================
   
    if distr == "IG":    
        output_filename = f"IG_{datetime.now().strftime('%d.%m.%Y_%H.%M')} {estimation_method}.out"
        tee = Tee(output_filename)
        sys.stdout = tee

        print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data Name: {data_name}")
        print(f"Method: {estimation_method}")

        results = find_optimal_gamma_ig(data, censored, 0.0, 0.99, 1e-6, 50)

        print("=" * 80)
        print("ОЦЕНКА ПАРАМЕТРОВ ОБРАТНОГО НОРМАЛЬНОГО РАСПРЕДЕЛЕНИЯ")
        print("=" * 80)
        print(f"Количество образцов: {len(data)}")
        print(f"  Разрушено: {np.sum(censored == 0)}")
        print(f"  Цензурировано: {np.sum(censored == 1)}")
        print(f"Диапазон данных: [{np.min(data):.2f}, {np.max(data):.2f}]")
        print(f"Минимальное разрушенное значение: {np.min(data[censored == 0]):.2f}")
        print("=" * 80)
        print("ОЦЕНКИ ПАРАМЕТРОВ")
        print("=" * 80)
        print(f"\nОПТИМАЛЬНЫЙ ПОРОГ γ = {results['optimal_gamma']:.2f} циклов")
        print(f"  Число итераций: {results['gamma_search_iterations']}")
        print(f"  Сообщение: {results['gamma_search_message']}")
        print()
        print(f"  μ = {results['mu']:.2e} циклов")
        print(f"  log₁₀(μ) = {np.log10(results['mu']):.4f}")
        print(f"  λ = {results['lam']:.2e}")
        print(f"  log₁₀(λ) = {np.log10(results['lam']):.4f}")
        print(f"  Медиана = {results['median']:.2e} циклов")
        print(f"  log₁₀(медиана) = {np.log10(results['median']):.4f}")
        print(f"  Среднее = {results['mean']:.2e} циклов")
        print(f"  log₁₀(среднее) = {np.log10(results['mean']):.4f}")
        print(f"  Ст.откл. = {results['std']:.2e} циклов")
        print(f"  log₁₀(ст.откл.) = {np.log10(results['std']):.4f}")
        print(f"\nМИНИМИЗИРУЕМАЯ ФУНКЦИЯ:")
        print(f"  Q = {results['Q_min']:.12e}")
        print(f"  Число оценок функции: {results['nfev']}")
        print(f"  Успех: {results['success']}")
        print(f"  Сообщение: {results['message']}")
        # Ковариационная матрица
        #print("\n" + "=" * 80)
        #print("КОВАРИАЦИОННАЯ МАТРИЦА ПАРАМЕТРОВ:")
        #print("=" * 80)
        #cov = results['cov_matrix']
        #print(f"\n  Cov(μ,μ) = {cov[0,0]:.6e}")
        #print(f"  Cov(μ,λ) = {cov[0,1]:.6e}")
        #print(f"  Cov(λ,μ) = {cov[1,0]:.6e}")
        #print(f"  Cov(λ,λ) = {cov[1,1]:.6e}")
        # Проверка симметричности
        #if abs(cov[0,1] - cov[1,0]) > 1e-6:
        #    print(f"\n  ВНИМАНИЕ: Ковариационная матрица несимметрична!")
        #else:
        #    print(f"\n  Ковариационная матрица симметрична")
        #print(f"\n  Стандартные ошибки:")
        #print(f"    SE(μ) = {results['se_mu']:.6e} циклов (отн. {results['se_mu']/results#['mu']*100:.1f}%)")
        #print(f"    SE(λ) = {results['se_lam']:.6e} (отн. {results['se_lam']/results['lam']*100:.1f}%)")

#=============================================================================

    if distr == "W":
        output_filename = f"W_{datetime.now().strftime('%d.%m.%Y_%H.%M')}.out"
        tee = Tee(output_filename)
        sys.stdout = tee

        print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data Name: {data_name}")

        results = find_optimal_gamma_w(data, censored, 0.0, 0.99, 1e-6, 50)

        print("=" * 80)
        print("ОЦЕНКА ПАРАМЕТРОВ РАСПРЕДЕЛЕНИЯ ВЕЙБУЛЛА")
        print("=" * 80)
        print(f"Количество образцов: {len(data)}")
        print(f"  Разрушено: {np.sum(censored == 0)}")
        print(f"  Цензурировано: {np.sum(censored == 1)}")
        print(f"Диапазон данных: [{np.min(data):.2f}, {np.max(data):.2f}]")
        print(f"Минимальное разрушенное значение: {np.min(data[censored == 0]):.2f}")
        print()
        print("=" * 80)
        print("ОЦЕНКИ ПАРАМЕТРОВ")
        print("=" * 80)
        print(f"\n γ = {results['optimal_gamma']:.2f} циклов")
        print(f"  Поиск сошёлся: {results['gamma_search_converged']}")
        print(f"  Число итераций: {results['gamma_search_iterations']}")
        print(f"  Сообщение: {results['gamma_search_message']}")
        print(f"  shape (β) = {results['shape']:.8f}")
        print(f"  scale (η) = {results['scale']:.2f} циклов")
        print(f"  Q (c*c)   = {results['Q']:.15f}")
        median = results['median']
        print(f"  Медиана = {median:.2f} циклов")
        print(f"  log10(медиана) = {np.log10(median):.4f}")
        mean_val = results['mean']
        print(f"  Среднее = {mean_val:.2f} циклов")
        print(f"  log10(среднее) = {np.log10(mean_val):.4f}")
        std_val = results['std']
        print(f"  Ст.откл. = {std_val:.2f} циклов")
        print(f"  log10(ст.откл.) = {np.log10(std_val):.4f}")
        print()
        # Ковариационная матрица
        print("=" * 80)
        print("КОВАРИАЦИОННАЯ МАТРИЦА ПАРАМЕТРОВ")
        print("=" * 80)
        cov = results['cov_matrix']
        print(f"\n  Cov(β,β)     = {cov[0,0]:.8f}")
        print(f"  Cov(β,η)     = {cov[0,1]:.8f}")
        print(f"  Cov(η,β)     = {cov[1,0]:.8f}")
        print(f"  Cov(η,η)     = {cov[1,1]:.8f}")
        se_shape = results['se_shape']
        se_scale = results['se_scale']
        print(f"\nСТАНДАРТНЫЕ ОШИБКИ ПАРАМЕТРОВ:")
        print(f"  SE(β)   = {se_shape:.6f}  (отн. {se_shape/results['shape']*100:.1f}%)")
        print(f"  SE(η)   = {se_scale:.2f} циклов  (отн. {se_scale/results['scale']*100:.1f}%)")
        print()

    # Оценка Каплана-Мейера
    print("\n" + "=" * 90)
    print(f"\nОценка Каплана-Мейера:")
    print("\n" + "=" * 90)
    print()
   
    km_times, km_survival, km_cdf, km_var, km_se = kaplan_meier(data, censored)
    
    if distr == "BS": 
        theo_cdf = bs_cdf(km_times, results['alpha'], results['beta'], results['loc'])
        plot_cdf_kaplan_meier_bs(results)

    if distr == "IG": 
        theo_cdf = invgauss_cdf(km_times, results['mu'], results['lam'], results['loc'])
        plot_cdf_kaplan_meier_ig(results)

    if distr == "W": 
        theo_cdf = weibull_cdf(km_times, results['shape'], results['scale'], results['loc'])
        plot_cdf_kaplan_meier_weibull(results)

    print(f"{'№':<4} {'Наработка':>18} {'F(t)':>12} {'Ftheo(t)':>12}"
          f"{'log10(N)':>12} {'log10_low':^12} {'log10_up':^12}")
    print("-" * 90)
    for i in range(len(km_times)):
        p_val = km_cdf[i]
        q_val = np.log10(km_times[i])
        ci = delta_method_km_quantile_ci(km_times, km_survival, km_var, p_val)
        print(f"{i+1:<4} {km_times[i]:18.2f} {km_cdf[i]:12.5f} {theo_cdf[i]:12.5f}"
              f"{np.log10(km_times[i]):12.7f} {ci['ci_lower_log10']:12.7f} {ci['ci_upper_log10']:12.7f}")

#====================Bootstrap======================================================

    if boots == "NO": return

    if distr == "BS": 
        print("\n" + "=" * 80)
        print("ПАРАМЕТРИЧЕСКИЙ БУТСТРЕП ДЛЯ BS-РАСПРЕДЕЛЕНИЯ")
        print("=" * 80)
        print(f"  Число бутстреп-итераций: 500")
        print(f"  Уровень доверия: 95%")
    
        bootstrap_results = []
    
        for p in quantiles:
            result = parametric_bootstrap_bs(
            data=data,
            censored=censored,
            alpha_hat=results['alpha'],
            beta_hat=results['beta'],
            gamma=results['optimal_gamma'],
            p=p,
            n_bootstrap=500,
            conf_level=0.95,
            random_seed=42
            )
            bootstrap_results.append(result)
    
        print("\n" + "=" * 80)
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ДОВЕРИТЕЛЬНЫХ ИНТЕРВАЛОВ (95%)")
        print("=" * 80)
        print(f"\n{'p':^6} {'Квантиль (циклы)':^20} {'log₁₀(N)':^12} {'95% ДИ log₁₀':^28} {'Ширина':^10}")
        print("-" * 80)
    
        for res in bootstrap_results:
            print(f"{res.p:^6.2f} {res.quantile_original:20.2e} {res.quantile_log10:12.6f} "
              f"[{res.ci_lower_log10:8.4f}, {res.ci_upper_log10:8.4f}] {res.ci_width_log10:10.4f}")

#=======================================================================================================

    if distr == "IG": 
        print("\n" + "=" * 80)
        print("ПАРАМЕТРИЧЕСКИЙ БУТСТРЕП ДЛЯ IG-РАСПРЕДЕЛЕНИЯ")
        print("=" * 80)
        print(f"  Число бутстреп-итераций: 500")
        print(f"  Уровень доверия: 95%")

        bootstrap_results = []
        gamma_opt = results['optimal_gamma']
        data_corrected = data - gamma_opt

        for p in quantiles:
            result = parametric_bootstrap_ig(
                data=data_corrected, censored=censored, mu_hat=results['mu'], lam_hat=results['lam'],
                gamma=gamma_opt, p=p, n_bootstrap=500, conf_level=0.95, random_seed=42)
        
            bootstrap_results.append(result)

        print("\n" + "=" * 80)
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ДОВЕРИТЕЛЬНЫХ ИНТЕРВАЛОВ (95%)")
        print("=" * 80)
        print(f"\n{'p':^6} {'Квантиль (циклы)':^20} {'log₁₀(N)':^12} {'95% ДИ log₁₀':^28} {'Ширина':^10}")
        print("-" * 80)

        for res in bootstrap_results:
            print(f"{res.p:^6.2f} {res.quantile_original:20.2e} {res.quantile_log10:12.6f} "
              f"[{res.ci_lower_log10:8.4f}, {res.ci_upper_log10:8.4f}] {res.ci_width_log10:10.4f}")

#==================================================================================================

    if distr == "W":
        print("\n" + "=" * 80)
        print("ПАРАМЕТРИЧЕСКИЙ БУТСТРЕП ДЛЯ РАСПРЕДЕЛЕНИЯ ВЕЙБУЛЛА")
        print("=" * 80)
        print(f"  Число бутстреп-итераций: 500")
        print(f"  Уровень доверия: 95%")

        bootstrap_results = []
        gamma_opt = results['optimal_gamma']
        data_corrected = data - gamma_opt

        for p in quantiles:
            result = parametric_bootstrap_w(
            data=data_corrected,
            censored=censored,
            shape_hat=results['shape'],
            scale_hat=results['scale'],
            gamma=gamma_opt,
            p=p,
            n_bootstrap=500,
            conf_level=0.95,
            random_seed=42
            )
            bootstrap_results.append(result)

        print("\n" + "=" * 80)
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ДОВЕРИТЕЛЬНЫХ ИНТЕРВАЛОВ (95%)")
        print("=" * 80)
        print(f"\n{'p':^6} {'Квантиль (циклы)':^20} {'log₁₀(N)':^12} {'95% ДИ log₁₀':^28} {'Ширина':^10}")
        print("-" * 80)

        for res in bootstrap_results:
            print(f"{res.p:^6.2f} {res.quantile_original:20.2e} {res.quantile_log10:12.6f} "
              f"[{res.ci_lower_log10:8.4f}, {res.ci_upper_log10:8.4f}] {res.ci_width_log10:10.4f}")


if __name__ == "__main__":
    main()