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
    random_seed: int = 42) -> BootstrapResult:
    
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
    
    for b in range(n_bootstrap):
        # Генерация выборки (в скорректированной шкале, loc=0)
        u = np.random.uniform(0, 1, n)
        t_adj_full = weibull_ppf(u, shape_hat, scale_hat, 0.0)
        t_adj_full.sort()
        
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
    
    print(f"    95% ДИ log: [{ci_lower_log10:.6f}, {ci_upper_log10:.6f}]")
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
    

def process_estimation_w() -> bool:

    inp_dir="Inp"
    out_dir="Out"
    inp_file="estimation_w.json"
    out_file="estimation_w.out" 
    fout=open(out_dir+"/"+out_file,'w')

#=======Загрузка данных==================================================

    with open(inp_dir+"/"+inp_file, 'r', encoding='windows-1251') as f:params_dict = json.load(f)
    data_params = CensoredData.from_dict(params_dict)
    data_name = data_params.data_name
    distr = data_params.distr_type

    global estimation_method
    estimation_method=params_dict.get('estimation_method','analytical')

    boots = data_params.bootstrap
    quantiles = data_params.quantiles
    data = np.array(data_params.data)
    censored = np.array(data_params.censored)

#=============================================================================

    if distr == "W":
        output_filename = f"W_{datetime.now().strftime('%d.%m.%Y_%H.%M')}.out"

        print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",file=fout)
        print(f"Data Name: {data_name}",file=fout)

        results = find_optimal_gamma_w(data, censored, 0.0, 0.99, 1e-6, 50)

        print("=" * 80,file=fout)
        print("ОЦЕНКА ПАРАМЕТРОВ РАСПРЕДЕЛЕНИЯ ВЕЙБУЛЛА",file=fout)
        print("=" * 80,file=fout)
        print(f"Количество образцов: {len(data)}",file=fout)
        print(f"  Разрушено: {np.sum(censored == 0)}",file=fout)
        print(f"  Цензурировано: {np.sum(censored == 1)}",file=fout)
        print(f"Диапазон данных: [{np.min(data):.2f}, {np.max(data):.2f}]",file=fout)
        print(f"Минимальное разрушенное значение: {np.min(data[censored == 0]):.2f}",file=fout)
        print("=" * 80,file=fout)
        print("ОЦЕНКИ ПАРАМЕТРОВ",file=fout)
        print("=" * 80,file=fout)
        print(f"\n gamma={results['optimal_gamma']:.2f} циклов",file=fout)
        print(f"  Поиск сошёлся: {results['gamma_search_converged']}",file=fout)
        print(f"  Число итераций: {results['gamma_search_iterations']}",file=fout)
        print(f"  Сообщение: {results['gamma_search_message']}",file=fout)
        print(f"  shape (beta) = {results['shape']:.8f}",file=fout)
        print(f"  scale (eta) = {results['scale']:.2f} циклов",file=fout)
        print(f"  Q (c*c)   = {results['Q']:.15f}",file=fout)
        median = results['median']
        print(f"  Медиана = {median:.2f} циклов",file=fout)
        print(f"  log10(медиана) = {np.log10(median):.4f}",file=fout)
        mean_val = results['mean']
        print(f"  Среднее = {mean_val:.2f} циклов",file=fout)
        print(f"  log10(среднее) = {np.log10(mean_val):.4f}",file=fout)
        std_val = results['std']
        print(f"  Ст.откл. = {std_val:.2f} циклов",file=fout)
        print(f"  log10(ст.откл.) = {np.log10(std_val):.4f}",file=fout)
        # Ковариационная матрица
        print("=" * 80,file=fout)
        print("КОВАРИАЦИОННАЯ МАТРИЦА ПАРАМЕТРОВ",file=fout)
        print("=" * 80,file=fout)
        cov = results['cov_matrix']
        print(f"\n  Cov(beta,beta)     = {cov[0,0]:.8f}",file=fout)
        print(f"  Cov(beta,eta)     = {cov[0,1]:.8f}",file=fout)
        print(f"  Cov(eta,beta)     = {cov[1,0]:.8f}",file=fout)
        print(f"  Cov(eta,eta)     = {cov[1,1]:.8f}",file=fout)
        se_shape = results['se_shape']
        se_scale = results['se_scale']
        print(f"\nСТАНДАРТНЫЕ ОШИБКИ ПАРАМЕТРОВ:",file=fout)
        print(f"  SE(beta)   = {se_shape:.6f}  (отн. {se_shape/results['shape']*100:.1f}%)",file=fout)
        print(f"  SE(eta)   = {se_scale:.2f} циклов  (отн. {se_scale/results['scale']*100:.1f}%)",file=fout)

    # Оценка Каплана-Мейера
    print("\n" + "=" * 90,file=fout)
    print(f"\nОценка Каплана-Мейера:",file=fout)
    print("\n" + "=" * 90,file=fout)
   
    km_times, km_survival, km_cdf, km_var, km_se = kaplan_meier(data, censored)

    if distr == "W": 
        theo_cdf = weibull_cdf(km_times, results['shape'], results['scale'], results['loc'])
        plot_cdf_kaplan_meier_weibull(results)

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

    if distr == "W":
        print("\n" + "=" * 80,file=fout)
        print("ПАРАМЕТРИЧЕСКИЙ БУТСТРЕП ДЛЯ РАСПРЕДЕЛЕНИЯ ВЕЙБУЛЛА",file=fout)
        print("=" * 80,file=fout)
        print(f"  Число бутстреп-итераций: 500",file=fout)
        print(f"  Уровень доверия: 95%",file=fout)

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
            random_seed=42)
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
    process_estimation_w()

