import numpy as np
import json
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
    
    #print(f"  Диапазон поиска γ: [{min_gamma:.2f}, {max_gamma:.2f}]")
    
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
        t_adj_full.sort()
        
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
# Визуализация
# ================================================================

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



# ================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ================================================================
    

def process_estimation_ig_num() -> bool:

    inp_dir="Inp"
    out_dir="Out"
    inp_file="estimation_ig_num.json"
    out_file="estimation_ig_num.out" 
    fout=open(out_dir+"/"+out_file,'w')

#=======Загрузка данных==================================================

    with open(inp_dir+"/"+inp_file, 'r', encoding='windows-1251') as f:params_dict = json.load(f)
    data_params = CensoredData.from_dict(params_dict)
    data_name = data_params.data_name
    distr = data_params.distr_type

    global estimation_method
    estimation_method='numerical'

    boots = data_params.bootstrap
    quantiles = data_params.quantiles
    data = np.array(data_params.data)
    censored = np.array(data_params.censored)

#=============================================================================

    if distr == "IG":    

        print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",file=fout)
        print(f"Data Name: {data_name}",file=fout)
        print(f"Method: {estimation_method}",file=fout)

        results = find_optimal_gamma_ig(data, censored, 0.0, 0.99, 1e-6, 50)

        print("=" * 80,file=fout)
        print("ОЦЕНКА ПАРАМЕТРОВ ОБРАТНОГО НОРМАЛЬНОГО РАСПРЕДЕЛЕНИЯ",file=fout)
        print("=" * 80,file=fout)
        print(f"Количество образцов: {len(data)}",file=fout)
        print(f"  Разрушено: {np.sum(censored == 0)}",file=fout)
        print(f"  Цензурировано: {np.sum(censored == 1)}",file=fout)
        print(f"Диапазон данных: [{np.min(data):.2f}, {np.max(data):.2f}]",file=fout)
        print(f"Минимальное разрушенное значение: {np.min(data[censored == 0]):.2f}",file=fout)
        print("=" * 80,file=fout)
        print("ОЦЕНКИ ПАРАМЕТРОВ",file=fout)
        print("=" * 80,file=fout)
        print(f"\nОПТИМАЛЬНЫЙ ПОРОГ gamma = {results['optimal_gamma']:.2f} циклов",file=fout)
        print(f"  Число итераций: {results['gamma_search_iterations']}",file=fout)
        print(f"  Сообщение: {results['gamma_search_message']}",file=fout)
        print(f"  mu = {results['mu']:.2e} циклов",file=fout)
        print(f"  lg(mu) = {np.log10(results['mu']):.4f}",file=fout)
        print(f"  lambda = {results['lam']:.2e}",file=fout)
        print(f"  lg(lambda) = {np.log10(results['lam']):.4f}",file=fout)
        print(f"  Медиана = {results['median']:.2e} циклов",file=fout)
        print(f"  lg(медиана) = {np.log10(results['median']):.4f}",file=fout)
        print(f"  Среднее = {results['mean']:.2e} циклов",file=fout)
        print(f"  lg(среднее) = {np.log10(results['mean']):.4f}",file=fout)
        print(f"  Ст.откл. = {results['std']:.2e} циклов",file=fout)
        print(f"  lg(ст.откл.) = {np.log10(results['std']):.4f}",file=fout)
        print(f"\nМИНИМИЗИРУЕМАЯ ФУНКЦИЯ:",file=fout)
        print(f"  Q = {results['Q_min']:.12e}",file=fout)
        print(f"  Число оценок функции: {results['nfev']}",file=fout)
        print(f"  Успех: {results['success']}",file=fout)
        print(f"  Сообщение: {results['message']}",file=fout)

    # Оценка Каплана-Мейера
    print("\n" + "=" * 90,file=fout)
    print(f"\nОценка Каплана-Мейера:",file=fout)
    print("\n" + "=" * 90,file=fout)
   
    km_times, km_survival, km_cdf, km_var, km_se = kaplan_meier(data, censored)
    
    if distr == "IG": 
        theo_cdf = invgauss_cdf(km_times, results['mu'], results['lam'], results['loc'])
        plot_cdf_kaplan_meier_ig(results)


    print(f"{'№':<4} {'Наработка':>18} {'F(t)':>12} {'Ftheo(t)':>12}"
          f"{'log10(N)':>12} {'log10_low':^12} {'log10_up':^12}",file=fout)
    print("-" * 90,file=fout)
    for i in range(len(km_times)):
        p_val = km_cdf[i]
        q_val = np.log10(km_times[i])
        ci = delta_method_km_quantile_ci(km_times, km_survival, km_var, p_val)
        print(f"{i+1:<4} {km_times[i]:18.2f} {km_cdf[i]:12.5f} {theo_cdf[i]:12.5f}"
              f"{np.log10(km_times[i]):12.7f} {ci['ci_lower_log10']:12.7f} {ci['ci_upper_log10']:12.7f}",file=fout)

#====================Bootstrap======================================================

    if boots == "NO": 
        fout.close()
        return True
   

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

        print("\n" + "=" * 80,file=fout)
        print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ДОВЕРИТЕЛЬНЫХ ИНТЕРВАЛОВ (95%)",file=fout)
        print("=" * 80,file=fout)
        print(f"\n{'p':^6} {'Квантиль (циклы)':^20} {'lg(N)':^12} {'95% ДИ lg':^28} {'Ширина':^10}",file=fout)
        print("-" * 80,file=fout)

        for res in bootstrap_results:
            print(f"{res.p:^6.2f} {res.quantile_original:20.2e} {res.quantile_log10:12.6f} "
              f"[{res.ci_lower_log10:8.4f}, {res.ci_upper_log10:8.4f}] {res.ci_width_log10:10.4f}",file=fout)

    fout.close()  
    return True

if __name__ == "__main__":
    process_estimation_ig_num()

