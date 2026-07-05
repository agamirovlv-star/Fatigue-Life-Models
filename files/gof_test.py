"""
================================================================================
УНИВЕРСАЛЬНЫЕ GOF-ТЕСТЫ ДЛЯ ЦЕНЗУРИРОВАННЫХ И ПОЛНЫХ ДАННЫХ
================================================================================

Поддерживаемые распределения:
    BS       - Бирнбаума-Сандерса (Birnbaum-Saunders)
    IG       - Обратное нормальное (Inverse Gaussian)
    Weibull  - Вейбулла (Weibull)

Поддерживаемые критерии:
    ks       - Колмогорова-Смирнова
    cvm      - Крамера-фон Мизеса
    ad       - Андерсона-Дарлинга (ОСНОВНОЙ КРИТЕРИЙ)
    shapiro  - Шапиро-Уилка (ТОЛЬКО ДЛЯ ПОЛНЫХ ВЫБОРОК)
    all      - Все доступные критерии

ДЛЯ ВСЕХ ТИПОВ ДАННЫХ используется параметрический бутстреп с переоценкой параметров.
ДЛЯ ПОЛНЫХ ВЫБОРОК критерий shapiro доступен дополнительно к ks/cvm/ad.

================================================================================
ФАЙЛЫ:
    gof_test.json                - настройки
    estimation.json - данные
    gof_test.out                  - выходной файл
================================================================================
"""

import numpy as np
import json
from datetime import datetime
from typing import Callable, Dict, Tuple, Optional, List
import warnings
import estimation
warnings.filterwarnings('ignore')

# ===================== Импорт модулей с оценкой параметров из estimation.py ====================
from estimation import (
    fit_bs_fixed_loc,
    bs_cdf,
    bs_ppf,
    fit_ig_fixed_loc,
    invgauss_cdf,
    invgauss_ppf_1 as ig_ppf,
    fit_weibull_fixed_loc,
    weibull_cdf,
    weibull_ppf,
    find_optimal_gamma_bs,
    find_optimal_gamma_ig,
    find_optimal_gamma_w,
    kaplan_meier
)


# ================================================================
# Критические значения для тестов
# ================================================================

# Критические значения для теста Колмогорова-Смирнова (приближённые)
KS_CRITICAL = {
    0.20: 1.07,
    0.15: 1.14,
    0.10: 1.22,
    0.05: 1.36,
    0.025: 1.48,
    0.01: 1.63,
    0.005: 1.73,
    0.001: 1.95
}

# Критические значения для теста Крамера-фон Мизеса (для цензурированных данных)
CVM_CRITICAL = {
    0.10: 0.347,
    0.05: 0.461,
    0.025: 0.581,
    0.01: 0.743
}

# Критические значения для теста Андерсона-Дарлинга (для цензурированных данных)
AD_CRITICAL = {
    0.10: 1.933,
    0.05: 2.492,
    0.025: 3.070,
    0.01: 3.857
}

#================================================================================

def get_p_value_range(statistic: float, critical_values: Dict[str, float]) -> str:
    """
    Определяет диапазон p-value на основе критических значений.
    critical_values: словарь {p_value: critical_value}
    """
    if np.isnan(statistic):
        return "не определён"
    
    # Сортируем по возрастанию p-value
    sorted_items = sorted(critical_values.items())
    
    # Если статистика меньше наименьшего критического значения
    if statistic < sorted_items[0][1]:
        return f"> {sorted_items[0][0]}"
    
    # Если статистика больше наибольшего критического значения
    if statistic > sorted_items[-1][1]:
        return f"< {sorted_items[-1][0]}"
    
    # Находим диапазон
    for i in range(len(sorted_items) - 1):
        if sorted_items[i][1] <= statistic < sorted_items[i+1][1]:
            return f"{sorted_items[i+1][0]} - {sorted_items[i][0]}"
    
    return "не определён"

#================================================================================

def format_critical_values(critical_values: Dict[str, float], test_name: str, statistic: float) -> List[str]:
    """Форматирует вывод критических значений и определяет диапазон p-value"""
    lines = []
    lines.append(f"  Критические значения для {test_name}:")
    lines.append("    alpha (уровень значимости)-критическое значение")
    lines.append("    " + "-" * 35)
    
    for alpha, cv in sorted(critical_values.items()):
        marker = "*" if statistic >= cv else ""
        lines.append(f"    {alpha:.3f}-{cv:.3f}{marker}")
    
    p_range = get_p_value_range(statistic, critical_values)
    lines.append(f"  Статистика = {statistic:.6f}")
    lines.append(f"  Диапазон p-value (приближённый): {p_range}")
    return lines

# ================================================================
# Модифицированные тесты для цензурированных данных
# ================================================================

def ks_test_censored(data: np.ndarray, censored: np.ndarray,
                      cdf_func: Callable, params: Dict) -> Tuple[float, float]:
    """Тест Колмогорова-Смирнова для цензурированных данных"""
    # Используем импортированную функцию kaplan_meier
    km_times, _, km_cdf, _, _ = kaplan_meier(data, censored)
    
    if len(km_times) == 0:
        return np.nan, np.nan
    
    theo_cdf = cdf_func(km_times, **params)
    D_n = np.max(np.abs(km_cdf - theo_cdf))
    n_eff = np.sum(censored == 0)
    
    # Масштабирование статистики для сравнения с критическими значениями
    D_n_scaled = D_n * np.sqrt(n_eff)
    
    # Приближённый p-value
    p_value = 2 * np.exp(-2 * n_eff * D_n * D_n) if D_n > 0 else 1.0
    p_value = min(1.0, p_value)
    
    return D_n_scaled, p_value

#================================================================================

def cvm_test_censored(data: np.ndarray, censored: np.ndarray,
                       cdf_func: Callable, params: Dict) -> Tuple[float, float]:
    """Тест Крамера-фон Мизеса для цензурированных данных"""
    # Получаем времена разрушений и эмпирическую CDF
    km_times, _, km_cdf, _, _ = kaplan_meier(data, censored)
    
    if len(km_times) < 2:
        return np.nan, np.nan
    
    # Теоретическая CDF в точках Каплана-Мейера
    theo_cdf = cdf_func(km_times, **params)
    n_fail = np.sum(censored == 0)
    
    # Сортируем по теоретической CDF
    sort_idx = np.argsort(theo_cdf)
    theo_sorted = theo_cdf[sort_idx]
    km_sorted = km_cdf[sort_idx]
    
    # Расчет статистики
    W2 = 0.0
    n = len(theo_sorted)
    
    for i in range(n):
        # Разность между эмпирической и теоретической CDF
        diff = km_sorted[i] - theo_sorted[i]
        W2 += diff * diff
    
    # Добавляем поправку для интервалов между точками
    for i in range(n - 1):
        # Вклад интервала между точками
        interval_len = theo_sorted[i+1] - theo_sorted[i]
        if interval_len > 0:
            mid_diff = (km_sorted[i] + km_sorted[i+1])/2 - (theo_sorted[i] + theo_sorted[i+1])/2
            W2 += mid_diff * mid_diff * interval_len / 3
    
    # Масштабирование статистики (классическая формула)
    W2 = W2 * n_fail
    
    # Поправка для малых выборок
    if n_fail < 100:
        W2 = W2 * (1 + 0.5 / n_fail)
    
    # Вычисление p-value на основе критических значений
    p_value = None
    for alpha, cv in CVM_CRITICAL.items():
        if W2 < cv:
            p_value = alpha
            break
    if p_value is None:
        p_value = 0.01
    if W2 < list(CVM_CRITICAL.values())[0]:
        p_value = 0.10
    
    return W2, p_value

#================================================================================

def ad_test_censored(data: np.ndarray, censored: np.ndarray,
                      cdf_func: Callable, params: Dict) -> Tuple[float, float]:
    """Тест Андерсона-Дарлинга для цензурированных данных"""
    failure_mask = censored == 0
    failure_data = data[failure_mask]
    
    if len(failure_data) < 2:
        return np.nan, np.nan
    
    u = cdf_func(failure_data, **params)
    u = np.sort(u)
    u = np.clip(u, 1e-10, 1 - 1e-10)
    
    n_r = len(u)
    
    A2 = -n_r
    for j in range(1, n_r + 1):
        A2 -= (2*j - 1) / n_r * (np.log(u[j-1]) + np.log(1 - u[n_r - j]))
    
    A2_corrected = A2 * (1 + 0.6 / n_r)
    
    p_value = None
    for alpha, cv in AD_CRITICAL.items():
        if A2_corrected < cv:
            p_value = alpha
            break
    if p_value is None:
        p_value = 0.01
    if A2_corrected < list(AD_CRITICAL.values())[0]:
        p_value = 0.10
    
    return A2_corrected, p_value


# ================== Тест Шапиро-Уилка для BS-распределения (полная выборка) ===============

def shapiro_test_bs(data: np.ndarray, params: Dict) -> Tuple[float, float]:
    from scipy.stats import shapiro
    
    alpha = params['alpha']
    beta = params['beta']
    loc = params.get('loc', 0)
    
    x_adj = np.maximum(data - loc, 1e-10)
    z = (np.sqrt(x_adj / beta) - np.sqrt(beta / x_adj)) / alpha
    
    statistic, p_value = shapiro(z)
    return statistic, p_value

#================================================================================

def shapiro_test_ig(data: np.ndarray, params: Dict) -> Tuple[float, float]:
    """Тест Шапиро-Уилка для IG-распределения (полная выборка)"""
    from scipy.stats import shapiro
    
    mu = params['mu']
    lam = params['lam']
    loc = params.get('loc', 0)
    
    x_adj = data - loc
    y = np.abs((x_adj - mu) / np.sqrt(x_adj))
    statistic, p_value = shapiro(y)
    return statistic, p_value

#================================================================================

def shapiro_test_weibull(data: np.ndarray, params: Dict) -> Tuple[float, float]:
    """Тест Шапиро-Уилка для распределения Вейбулла (полная выборка)"""
    from scipy.stats import shapiro
    
    loc = params.get('loc', 0)
    y = np.log(data - loc)
    statistic, p_value = shapiro(y)
    return statistic, p_value


# ================================================================
# Параметрический бутстреп (для всех тестов)
# ================================================================

def parametric_bootstrap_gof(data: np.ndarray, censored: np.ndarray,
                               fit_func: Callable,
                               cdf_func: Callable,
                               ppf_func: Callable,
                               test_func: Callable,
                               dist_name: str,
                               fixed_loc: float,
                               n_bootstrap: int = 999,
                               random_seed: int = 42) -> Tuple[float, int, float, float, float, np.ndarray]:
    """
    Параметрический бутстреп для GOF-тестов.
    Возвращает (p-value, число успешных итераций, наблюдаемая статистика,
                среднее бутстреп-статистик, стандартное отклонение, массив бутстреп-статистик)
    """
    np.random.seed(random_seed)
    
    fit_result = fit_func(data, censored, fixed_loc)
    
    if dist_name == 'BS':
        params = {'alpha': fit_result['alpha'], 'beta': fit_result['beta'], 'loc': fixed_loc}
    elif dist_name == 'IG':
        params = {'mu': fit_result['mu'], 'lam': fit_result['lam'], 'loc': fixed_loc}
    elif dist_name == 'Weibull':
        params = {'shape': fit_result['shape'], 'scale': fit_result['scale'], 'loc': fixed_loc}
    else:
        raise ValueError(f"Unknown distribution: {dist_name}")
    
    T_obs, _ = test_func(data, censored, cdf_func, params)
    
    T_bootstrap = []
    n = len(data)
    data_corrected = data - fixed_loc
    
    for b in range(n_bootstrap):
        u = np.random.uniform(0, 1, n)
        
        if dist_name == 'BS':
            full_sample = ppf_func(u, params['alpha'], params['beta'], 0.0)
        elif dist_name == 'IG':
            full_sample = ppf_func(u, params['mu'], params['lam'], 0.0)
        elif dist_name == 'Weibull':
            full_sample = ppf_func(u, params['shape'], params['scale'], 0.0)
        else:
            continue
        
        bootstrap_sample = full_sample.copy()
        for i in range(n):
            if censored[i] == 1 and full_sample[i] > data_corrected[i]:
                bootstrap_sample[i] = data_corrected[i]
        
        try:
            boot_result = fit_func(bootstrap_sample, censored, 0.0)
            
            if dist_name == 'BS':
                boot_params = {'alpha': boot_result['alpha'], 'beta': boot_result['beta'], 'loc': 0.0}
            elif dist_name == 'IG':
                boot_params = {'mu': boot_result['mu'], 'lam': boot_result['lam'], 'loc': 0.0}
            elif dist_name == 'Weibull':
                boot_params = {'shape': boot_result['shape'], 'scale': boot_result['scale'], 'loc': 0.0}
            else:
                continue
            
            T_b, _ = test_func(bootstrap_sample, censored, cdf_func, boot_params)
            T_bootstrap.append(T_b)
            
        except Exception:
            continue
    
    n_success = len(T_bootstrap)
    if n_success == 0:
        return np.nan, 0, T_obs, np.nan, np.nan, np.array([])
    
    T_bootstrap_arr = np.array(T_bootstrap)
    T_mean = np.mean(T_bootstrap_arr)
    T_std = np.std(T_bootstrap_arr)
    
    p_value = (1 + np.sum(T_bootstrap_arr >= T_obs)) / (1 + n_success)
    return p_value, n_success, T_obs, T_mean, T_std, T_bootstrap_arr


# ================================================================
# Запуск GOF-тестов для выбранного распределения. Автоматическое определение типа выборки
# ================================================================

def run_gof_tests(data: np.ndarray, censored: np.ndarray,
                  distribution: str,
                  test_type: str = 'all', n_bootstrap: int = 999,
                  random_seed: int = 42) -> Dict:
    
    is_full_sample = np.sum(censored == 1) == 0
    
    # Поиск оптимального порога γ и оценка параметров в зависимости от распределения
    if distribution == 'BS':
        fit_result = find_optimal_gamma_bs(data, censored,0.0, None, 1e-6, 50)
        fixed_loc = fit_result['optimal_gamma']
        cdf_func = bs_cdf
        ppf_func = bs_ppf
        def fit_func(d,c,loc): return fit_bs_fixed_loc(d,c,loc)
    elif distribution == 'IG':
        fit_result = find_optimal_gamma_ig(data, censored, 0.0, 0.99, 1e-6, 50)
        fixed_loc = fit_result['optimal_gamma']
        cdf_func = invgauss_cdf
        ppf_func = ig_ppf
        def fit_func(d, c, loc): return fit_ig_fixed_loc(d, c, loc)
    elif distribution == 'Weibull':
        fit_result = find_optimal_gamma_w(data, censored, 0.0, 0.99, 1e-6, 50)
        fixed_loc = fit_result['optimal_gamma']
        cdf_func = weibull_cdf
        ppf_func = weibull_ppf
        def fit_func(d, c, loc): return fit_weibull_fixed_loc(d, c, loc)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # Формирование параметров для CDF
    if distribution == 'BS':
        params = {'alpha': fit_result['alpha'], 'beta': fit_result['beta'], 'loc': fixed_loc}
    elif distribution == 'IG':
        params = {'mu': fit_result['mu'], 'lam': fit_result['lam'], 'loc': fixed_loc}
    elif distribution == 'Weibull':
        params = {'shape': fit_result['shape'], 'scale': fit_result['scale'], 'loc': fixed_loc}
    
    # Определение доступных тестов в зависимости от типа выборки
    if is_full_sample:
        test_map = {
            'ks': [('Kolmogorov-Smirnov', ks_test_censored, KS_CRITICAL)],
            'cvm': [('Cramer-von Mises', cvm_test_censored, CVM_CRITICAL)],
            'ad': [('Anderson-Darling', ad_test_censored, AD_CRITICAL)],
            'shapiro': [('Shapiro-Wilk', None, None)],
            'all': [('Kolmogorov-Smirnov', ks_test_censored, KS_CRITICAL),
                    ('Cramer-von Mises', cvm_test_censored, CVM_CRITICAL),
                    ('Anderson-Darling', ad_test_censored, AD_CRITICAL),
                    ('Shapiro-Wilk', None, None)]
        }
    else:
        test_map = {
            'ks': [('Kolmogorov-Smirnov', ks_test_censored, KS_CRITICAL)],
            'cvm': [('Cramer-von Mises', cvm_test_censored, CVM_CRITICAL)],
            'ad': [('Anderson-Darling', ad_test_censored, AD_CRITICAL)],
            'all': [('Kolmogorov-Smirnov', ks_test_censored, KS_CRITICAL),
                    ('Cramer-von Mises', cvm_test_censored, CVM_CRITICAL),
                    ('Anderson-Darling', ad_test_censored, AD_CRITICAL)]
        }
    
    if test_type not in test_map:
        raise ValueError(f"Unknown test_type: {test_type}")
    
    tests = test_map[test_type]
    
    approx_results = {}
    bootstrap_results = {}
    critical_info = {}
    
    for name, test_func, critical_vals in tests:
        if test_func is not None:
            stat, p_val = test_func(data, censored, cdf_func, params)
            approx_results[name] = {'statistic': stat, 'p_value': p_val}
            
            # Сохраняем критическую информацию
            if critical_vals is not None:
                critical_info[name] = {
                    'critical_values': critical_vals,
                    'statistic': stat,
                    'p_range': get_p_value_range(stat, critical_vals)
                }
            
            p_val_boot, n_success, t_obs, t_mean, t_std, t_dist = parametric_bootstrap_gof(
               data, censored, fit_func, cdf_func, ppf_func,
               test_func, distribution, fixed_loc, n_bootstrap, random_seed
            )

            bootstrap_results[name] = {
              'p_value': p_val_boot,
              'n_success': n_success,
              'statistic_observed': t_obs,
              'statistic_mean': t_mean,
              'statistic_std': t_std
            }
        else:
            if distribution == 'BS':
                stat, p_val = shapiro_test_bs(data, params)
            elif distribution == 'IG':
                stat, p_val = shapiro_test_ig(data, params)
            elif distribution == 'Weibull':
                stat, p_val = shapiro_test_weibull(data, params)
            else:
                stat, p_val = np.nan, np.nan
            
            approx_results[name] = {'statistic': stat, 'p_value': p_val}
            bootstrap_results[name] = {
                'p_value': p_val,
                'n_success': 0,
                'statistic_observed': stat,
                'note': 'Shapiro-Wilk test does not use bootstrap'
            }
    
    return {
        'distribution': distribution,
        'test_type': test_type,
        'is_full_sample': is_full_sample,
        'fit_results': fit_result,
        'approx_results': approx_results,
        'bootstrap_results': bootstrap_results,
        'critical_info': critical_info,
        'params': params,
        'n_total': len(data),
        'n_fail': int(np.sum(censored == 0)),
        'n_cens': int(np.sum(censored == 1)),
        'fixed_loc': fixed_loc
    }


# ================================================================
# Функции форматирования вывода
# ================================================================

def format_parameter_output(distribution: str, fit_result: Dict,fixed_loc: float) -> List[str]:

    lines = []
    lines.append(f"  Оптимальный порог gamma={fixed_loc:.2f} циклов")
    
    if distribution == 'BS':
        lines.append(f"  alpha={fit_result['alpha']:.8f}")
        lines.append(f"  beta={fit_result['beta']:.2f}")
        lines.append(f"  lg(beta)={np.log10(fit_result['beta']):.4f}")
        lines.append(f"  Медиана ={fit_result['median']:.2f} циклов")
    elif distribution == 'IG':
        lines.append(f"  mu={fit_result['mu']:.2f} циклов (lg={np.log10(fit_result['mu']):.4f})")
        lines.append(f"  lambda={fit_result['lam']:.2f} (lg={np.log10(fit_result['lam']):.4f})")
        lines.append(f"  Медиана = {fit_result['median']:.2f} циклов")
    elif distribution == 'Weibull':
        lines.append(f"  shape(beta)={fit_result['shape']:.8f}")
        lines.append(f"  scale(eta)={fit_result['scale']:.2f} циклов")
        lines.append(f"  Медиана={fit_result['median']:.2f} циклов")
    return lines

#================================================================================

def format_test_output(approx_results: Dict, bootstrap_results: Dict,is_full_sample: bool) -> List[str]:

    lines = []
    lines.append(f" {'Тест':<25} {'Статистика':<15} {'p-value (прибл.)':<18} {'p-value (бутстреп)':<18}")
    lines.append("-" * 80)
    
    for name in approx_results.keys():
        approx = approx_results[name]
        boot = bootstrap_results[name]
        
        if name == 'Shapiro-Wilk':
            lines.append(f"{name:<25} {approx['statistic']:<15.6f} {approx['p_value']:<18.4f} {'(not in use )':<18}")
        else:
            lines.append(f"{name:<25} {approx['statistic']:<15.6f} {approx['p_value']:<18.4f} {boot['p_value']:<18.4f}")
            lines.append(f"{'':<25} {'':<15} {'бутстреп среднее:':<18} {boot['statistic_mean']:<18.6f}")
            lines.append(f"{'':<25} {'':<15} {'бутстреп std:':<18} {boot['statistic_std']:<18.6f}")
    
    return lines

#================================================================================

def format_critical_output(critical_info: Dict) -> List[str]:
    lines = []
    
    test_names = {
        'Kolmogorov-Smirnov': 'Колмогорова-Смирнова',
        'Cramer-von Mises': 'Крамера-фон Мизеса',
        'Anderson-Darling': 'Андерсона-Дарлинга'
    }
    
    for name, info in critical_info.items():
        rus_name = test_names.get(name, name)
        lines.append("\n" + "=" * 50)
        lines.append(f"КРИТИЧЕСКИЕ ЗНАЧЕНИЯ ДЛЯ ТЕСТА {rus_name.upper()}")
        lines.append("=" * 50)
        lines.append(f"  Статистика критерия: {info['statistic']:.6f}")
        lines.append(f"  Диапазон p-value (приближённый): {info['p_range']}")
        lines.append(f"  Критические значения:")
        lines.append(f"    {'alpha(уровень значимости)':<25}->{'критическое значение':<20}")
        lines.append("    " + "-" * 50)
        
        for alpha, cv in sorted(info['critical_values'].items()):
            marker ="-статистика > крит. значения" if info['statistic'] >= cv else ""
            lines.append(f"    {alpha:<25}-{cv:<20.4f}{marker}")
        
        lines.append(f" Интерпретация: статистика находится в диапазоне p-value {info['p_range']}")
    
    return lines


# ================Main==============================================

def process_gof_test() -> bool:

    """
    Основная функция:
    1. Читает настройки из config.json
    2. Читает данные из estimation.json
    3. Вызывает функции из estimation.py для поиска оптимального γ и оценки параметров
    4. Автоматически определяет наличие цензуры
    5. Запускает GOF-тесты с бутстрепом
    6. Для полных данных дополнительно доступен Shapiro-Wilk
    7. Выводит результаты в консоль и файл
    """


    inp_dir="Inp"
    out_dir="Out"
    inp_file="gof_test.json"
    estimation_file="estimation.json"
    out_file="gof_test.out" 
    fout=open(out_dir+"/"+out_file,'w')


    print("=" * 80,file=fout)
    print("УНИВЕРСАЛЬНЫЕ GOF-ТЕСТЫ ДЛЯ ПОЛНЫХ И ЦЕНЗУРИРОВАННЫХ ДАННЫХ",file=fout)
    print("=" * 80,file=fout)
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",file=fout)
    
    # 1. Загрузка конфигурации
    print("\n" + "-" * 40,file=fout)
    print("ЗАГРУЗКА НАСТРОЕК",file=fout)
    print("-" * 40,file=fout)
    
    with open(inp_dir+"/"+inp_file,'r', encoding='windows-1251') as f: config=json.load(f)
    test_type = config.get('test_type','all')
    n_bootstrap = config.get('n_bootstrap',500)
    random_seed = config.get('random_seed',42)
    
    print(f"  Критерий: {test_type}",file=fout)
    print(f"  Бутстреп-итераций: {n_bootstrap}",file=fout)
    print(f"  Random seed: {random_seed}",file=fout)

    # 2. Загрузка данных
    print("\n" + "-" * 40,file=fout)
    print("ЗАГРУЗКА ДАННЫХ",file=fout)
    print("-" * 40,file=fout)

    with open(inp_dir+"/"+estimation_file,'r',encoding='windows-1251') as f:  data_dict=json.load(f)

    data = np.array(data_dict['data'])
    censored = np.array(data_dict['censored'])
    distribution = data_dict.get('distr_type')
    estimation.estimation_method = data_dict.get('estimation_method', 'analytical')

    # Преобразование 'W' в 'Weibull' для единообразия
    if distribution == 'W': distribution = 'Weibull'

    print("=" * 80,file=fout)
    print("УНИВЕРСАЛЬНЫЕ GOF-ТЕСТЫ ДЛЯ ПОЛНЫХ И ЦЕНЗУРИРОВАННЫХ ДАННЫХ",file=fout)
    print("=" * 80,file=fout)
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",file=fout)
    
    n_total = len(data)
    n_fail = np.sum(censored == 0)
    n_cens = np.sum(censored == 1)
    is_full_sample = (n_cens == 0)
    
    print(f"  Распределение (из данных): {distribution}",file=fout)
    print(f"  Размер выборки: {n_total}",file=fout)
    print(f"  Разрушено: {n_fail}",file=fout)
    print(f"  Цензурировано: {n_cens}",file=fout)
    
    if is_full_sample:
        print("     ОБНАРУЖЕНА ПОЛНАЯ ВЫБОРКА (цензура отсутствует)",file=fout)
        print("     Доступен критерий Шапиро-Уилка",file=fout)
    else:
        print("     Обнаружена цензурированная выборка",file=fout)
        print("     Критерий Шапиро-Уилка НЕДОСТУПЕН",file=fout)
    
    # 3. Валидация
    valid_distributions = ['BS', 'IG', 'Weibull']
    
    if distribution not in valid_distributions:
        print(f"\nОШИБКА: distribution='{distribution}' не поддерживается!",file=fout)
        fout.close()
        return False
    
    # Для цензурированных данных проверяем test_type
    if not is_full_sample and test_type == 'shapiro':
        print(f"\nОШИБКА: Критерий 'shapiro' недоступен для цензурированных данных!",file=fout)
        print("Доступные критерии: ks, cvm, ad, all",file=fout)
        fout.close()
        return False
    
    # 4. Запуск тестов
    print("\n" + "-" * 40,file=fout)
    print("ВЫПОЛНЕНИЕ РАСЧЁТОВ",file=fout)
    print("-" * 40,file=fout)
    
    print("  Поиск оптимального gamma",file=fout)
    results = run_gof_tests(
        data=data,
        censored=censored,
        distribution=distribution,
        test_type=test_type,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed)
    
    # 5. Вывод результатов
    print("\n" + "-" * 40,file=fout)
    print("ОЦЕНКА ПАРАМЕТРОВ",file=fout)
    print("-" * 40,file=fout)
    
    param_lines = format_parameter_output(distribution, results['fit_results'], results['fixed_loc'])
    for line in param_lines:  print(f"\n{line}",file=fout)
    
    print("\n" + "-" * 40,file=fout)
    print("РЕЗУЛЬТАТЫ ТЕСТОВ",file=fout)
    print("-" * 40,file=fout)
    
    for name, approx in results['approx_results'].items():
        print(f"    Статистика = {approx['statistic']:.6f}",file=fout)
        print(f"    p-value (приближённый) = {approx['p_value']:.4f}",file=fout)
        
        boot = results['bootstrap_results'][name]
        if 'note' in boot:
            print(f"    {boot['note']}",file=fout)
        else:
            print(f"    Успешных бутстреп-итераций: {boot['n_success']}/{n_bootstrap}",file=fout)
            print(f"    p-value (бутстреп) = {boot['p_value']:.4f}",file=fout)
            print(f"    бутстреп среднее: {boot['statistic_mean']:.6f}",file=fout)
            print(f"    бутстреп std: {boot['statistic_std']:.6f}",file=fout)
    
    # Вывод критических значений
    if results['critical_info']:
        critical_lines = format_critical_output(results['critical_info'])
        for line in critical_lines: print(line,file=fout)
    
    print("\n" + "=" * 80,file=fout)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ",file=fout)
    print("=" * 80,file=fout)
    
    test_output_lines = format_test_output(
        results['approx_results'], 
        results['bootstrap_results'],
        results['is_full_sample']
    )
    for line in test_output_lines: print(line,file=fout)
    
    # Интерпретация
    print("\n" + "-" * 40,file=fout)
    print("ИНТЕРПРЕТАЦИЯ",file=fout)
    print("-" * 40,file=fout)
    
    alpha_level = 0.05
    print(f"\n  Уровень значимости alpha={alpha_level}",file=fout)
    
    # Основной критерий - Андерсона-Дарлинга
    if 'Anderson-Darling' in results['bootstrap_results']:
        ad_result = results['bootstrap_results']['Anderson-Darling']
        if 'note' not in ad_result:
            ad_p = ad_result['p_value']
            print(f"\n  ОСНОВНОЙ КРИТЕРИЙ (Андерсона-Дарлинга): p-value = {ad_p:.4f}",file=fout)
            if ad_p > alpha_level:
                print("   Распределение НЕ противоречит данным (гипотеза не отвергается)",file=fout)
            else:
                print("   Распределение ПРОТИВОРЕЧИТ данным (гипотеза отвергается)",file=fout)
    
    # Shapiro-Wilk (если есть)
    if 'Shapiro-Wilk' in results['bootstrap_results']:
        sw_result = results['bootstrap_results']['Shapiro-Wilk']
        sw_p = sw_result['p_value']
        print(f"\n  ДОПОЛНИТЕЛЬНЫЙ КРИТЕРИЙ (Шапиро-Уилка): p-value = {sw_p:.4f}",file=fout)
        if sw_p > alpha_level:
            print("   Распределение НЕ противоречит данным",file=fout)
        else:
            print("   Распределение ПРОТИВОРЕЧИТ данным",file=fout)
    
    # Итоговый вывод
    if 'Anderson-Darling' in results['bootstrap_results']:
        ad_result = results['bootstrap_results']['Anderson-Darling']
        if 'note' not in ad_result and ad_result['p_value'] > 0.05:
            print(f"\n  ВЫВОД: Распределение {distribution}",file=fout)
            print("         адекватно описывает экспериментальные данные.",file=fout)
            print("         Гипотеза о соответствии не отвергается.",file=fout)
        elif 'note' not in ad_result:
            print(f"\n  ВЫВОД: Распределение {distribution}",file=fout)
            print("         НЕ адекватно описывает экспериментальные данные",file=fout)
            print("         Гипотеза о соответствии отвергается",file=fout)
    
    fout.close()
    return True


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    process_gof_test()

