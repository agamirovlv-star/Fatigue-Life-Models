"""
================================================================================
GOF ТЕСТЫ ДЛЯ ПОЛНЫХ ВЫБОРОК - ТОЛЬКО МАННА
================================================================================

Критерий Манна-Шойера-Фертига для полных выборок.
Поддерживает:
    - BS (преобразование к нормальности)
    - Weibull (логарифмирование)
    - Normal (логнормальное - логарифмирование)

Тип распределения задаётся в mann_test.json (distr_type)
================================================================================
"""

import numpy as np
import json
import sys
from datetime import datetime
from scipy import stats
from scipy.special import betainc
from scipy.stats import beta
import warnings
warnings.filterwarnings('ignore')

# ===================== Импорт из estimation.py ====================
import estimation
from estimation import (
    fit_bs_fixed_loc,
    fit_weibull_fixed_loc,
    find_optimal_gamma_bs,
    find_optimal_gamma_w
)


# ================================================================
# 0. ПОРЯДКОВЫЕ СТАТИСТИКИ (из order.py)
# ================================================================

def ordern(r, n):
    """Математическое ожидание r-й порядковой статистики для N(0,1)"""
    p = 1
    pr = r / (n + 1)
    qr = 1 - pr
    xr = stats.norm.ppf(pr)
    dr = stats.norm.pdf(xr)
    xr1 = p / dr
    xr2 = xr * (p / dr) ** 2
    xr3 = (2 * xr ** 2 + 1) * (p / dr) ** 3
    xr4 = (6 * xr ** 3 + 7 * xr) * (p / dr) ** 4
    xr5 = (24 * xr ** 4 + 46 * xr ** 2 + 7) * (p / dr) ** 5
    xr6 = (120 * xr ** 5 + 326 * xr ** 3 + 127 * xr) * (p / dr) ** 6
    er = (xr + pr * qr * xr2 / (2 * (n + 2)) +
          pr * qr * ((qr - pr) * xr3 / 3 + pr * qr * xr4 / 8) / (n + 2) ** 2 +
          pr * qr * (-(qr - pr) * xr3 / 3 + ((qr - pr) ** 2 - pr * qr) * xr4 / 4 +
                     qr * pr * (qr - pr) * xr5 / 6 + (qr * pr) ** 2 * xr6 / 48) / (n + 2) ** 3)
    return er


def orderw(r, n):
    """Математическое ожидание r-й порядковой статистики для Gumbel"""
    pr = r / (n + 1)
    qr = 1 - pr
    xr = np.log(np.log(1 / (1 - pr)))
    xr1 = 1 / (np.log(1 / (1 - pr)) * (1 - pr))
    xr2 = xr1 * (1 / (1 - pr) - xr1)
    xr3 = xr2 ** 2 / xr1 + xr1 * (1 / (1 - pr) ** 2 - xr2)
    xr4 = (3 * xr1 * xr2 * xr3 - 2 * xr2 ** 3) / xr1 ** 2 + xr1 * (2 / (1 - pr) ** 3 - xr3)
    xr55 = (-12 * xr1 * xr2 ** 2 * xr3 + 3 * xr1 ** 2 * xr3 ** 2 +
            4 * xr1 ** 2 * xr2 * xr4 + 6 * xr2 ** 4)
    xr5 = xr55 / xr1 ** 3 + xr1 * (6 / (1 - pr) ** 4 - xr4)
    a1 = -12 * xr2 ** 3 * xr3 - 12 * xr1 * (2 * xr2 * xr3 ** 2 + xr2 ** 2 * xr4)
    b1 = 6 * xr1 * xr2 * xr3 ** 2 + 6 * xr1 ** 2 * xr3 * xr4
    c1 = 8 * xr1 * xr2 ** 2 * xr4 + 4 * xr1 ** 2 * (xr3 * xr4 + xr2 * xr5)
    d1 = 24 * xr2 ** 3 * xr3
    xr6 = ((xr1 ** 3 * (a1 + b1 + c1 + d1) - 3 * xr1 ** 2 * xr2 * xr55) / xr1 ** 6 +
           xr2 * (6 / (1 - pr) ** 4 - xr4) + xr1 * (24 / (1 - pr) ** 5 - xr5))
    er = (xr + pr * qr * xr2 / (2 * (n + 2)) +
          pr * qr * ((qr - pr) * xr3 / 3 + pr * qr * xr4 / 8) / (n + 2) ** 2 +
          pr * qr * (-(qr - pr) * xr3 / 3 + ((qr - pr) ** 2 - pr * qr) * xr4 / 4 +
                     qr * pr * (qr - pr) * xr5 / 6 + (qr * pr) ** 2 * xr6 / 48) / (n + 2) ** 3)
    return er


# ================================================================
# 1. КРИТЕРИЙ МАННА-ШОЙЕРА-ФЕРТИГА
# ================================================================

def mann_beta_approximation(n, alpha=0.05):

    nu1 = float((n - 1.0)/2.0)
    nu2 = float(n/2.0)
    w_crit = 1.-beta.ppf(1.0-alpha,nu1, nu2)
    return {
        'nu1': int(nu1),
        'nu2': int(nu2),
        'critical_value': w_crit,
        'distribution': f'Beta({int(nu1)}, {int(nu2)})'
    }


def mann_beta_pvalue(w_stat, n):
    """p-value через бета-распределение"""
    nu1 = float((n - 1) // 2)
    nu2 = float(n // 2)
    if nu1 == 0 or nu2 == 0:
        return np.nan
    return betainc(nu1, nu2, w_stat)


def mann_test_statistic(x, r):
    n = len(x)
    k1 = n // 2
    s2 = sum((x[i+1] - x[i]) / (r[i+1] - r[i]) for i in range(n-1))
    s3 = sum((x[i+1] - x[i]) / (r[i+1] - r[i]) for i in range(k1, n-1))
    return s3 / s2


def mann_mc_critical(n, dist_type, alpha=0.05, n_sim=100000, random_seed=42):
    np.random.seed(random_seed)
    if dist_type in ['normal', 'BS']:
        r = [ordern(i+1, n) for i in range(n)]
    else:  # weibull
        r = [orderw(i+1, n) for i in range(n)]
    k1 = n // 2
    wc = np.zeros(n_sim)
    
    for j in range(n_sim):
        if dist_type in ['normal', 'BS']:
            z = np.random.standard_normal(n)
        else:
            z = np.log(np.log(1.0 / (1.0 - np.random.random(n))))
        z = np.sort(z)
        s2 = sum((z[i+1] - z[i]) / (r[i+1] - r[i]) for i in range(n-1))
        s3 = sum((z[i+1] - z[i]) / (r[i+1] - r[i]) for i in range(k1, n-1))
        wc[j] = s3 / s2
        
        if (j + 1) % (n_sim // 20) == 0 or j == n_sim - 1:
            print_progress(j+1, n_sim, prefix='    MC Манна', suffix='итераций', length=30)
    
    wc = np.sort(wc)
    mcrit = wc[int(alpha * n_sim)]
    return mcrit, wc


def mann_mc_pvalue(w_stat, mc_dist, n_sim):
    for j in range(n_sim):
        if mc_dist[j] >= w_stat:
            return (j - 0.375 + 1.0) / (n_sim + 0.25)
    return 1.0


def mann_test(x, dist_type, alpha=0.05, n_sim=100000, use_mc=True, random_seed=42):
 
    n = len(x)
    x_sorted = np.sort(x)
    
    if dist_type in ['normal', 'BS']:
        r = np.array([ordern(i+1, n) for i in range(n)])
    else:  # weibull
        r = np.array([orderw(i+1, n) for i in range(n)])
    
    w_stat = mann_test_statistic(x_sorted, r)
    
    # Бета-аппроксимация
    beta_info = mann_beta_approximation(n, alpha)
    beta_crit = beta_info['critical_value']
    beta_pval = mann_beta_pvalue(w_stat, n)
    
    # Монте-Карло
    if use_mc:
        print(f"\n  Монте-Карло для теста Манна (n_sim={n_sim})...")
        mc_crit, mc_dist = mann_mc_critical(n, dist_type, alpha, n_sim, random_seed)
        mc_pval = mann_mc_pvalue(w_stat, mc_dist, n_sim)
    else:
        mc_crit = np.nan
        mc_pval = np.nan
    
    return {
        'statistic': w_stat,
        'critical_beta': beta_crit,
        'p_value_beta': beta_pval,
        'critical_mc': mc_crit,
        'p_value_mc': mc_pval,
        'n': n,
        'dist_type': dist_type,
        'beta_nu1': beta_info['nu1'],
        'beta_nu2': beta_info['nu2'],
        'beta_dist': beta_info['distribution']
    }


# ================================================================
# 2. ОСНОВНАЯ ФУНКЦИЯ ЗАПУСКА
# ================================================================

def prepare_data(data, distribution, gamma):
  
    n = len(data)
    data_adj = data - gamma
    censored = np.zeros(n)
    
    if distribution == 'BS':
        fit_result = fit_bs_fixed_loc(data, censored, gamma)
        alpha_hat = fit_result['alpha']
        beta_hat = fit_result['beta']
        z = (np.sqrt(data_adj / beta_hat) - np.sqrt(beta_hat / data_adj)) / alpha_hat
        dist_type = 'BS'
        params_str = f"α={alpha_hat:.6f}, β={beta_hat:.2f}, γ={gamma:.2f}"
        return z, dist_type, params_str, fit_result
        
    elif distribution == 'Weibull':
        fit_result = fit_weibull_fixed_loc(data, censored, gamma)
        shape_hat = fit_result['shape']
        scale_hat = fit_result['scale']
        z = np.log(data_adj)
        dist_type = 'weibull'
        params_str = f"shape={shape_hat:.6f}, scale={scale_hat:.2f}, γ={gamma:.2f}"
        return z, dist_type, params_str, fit_result
        
    elif distribution == 'Normal':
        # Логнормальное распределение (нормальное в логарифмической шкале)
        z = np.log(data_adj)
        dist_type = 'normal'
        params_str = f"логнормальное, γ={gamma:.2f}"
        # Для логнормального не нужна оценка параметров через estimation.py
        fit_result = {'distribution': 'Normal', 'gamma': gamma}
        return z, dist_type, params_str, fit_result
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def run_mann_test(data, distribution, gamma, n_sim=100000, use_mc=True, random_seed=42):
    z, dist_type, params_str, fit_result = prepare_data(data, distribution, gamma)
    result = mann_test(z, dist_type, 0.05, n_sim, use_mc, random_seed)
    result['distribution'] = distribution
    result['params'] = params_str
    result['gamma'] = gamma
    result['n'] = len(data)
    result['fit_result'] = fit_result
    return result


def print_progress(current, total, prefix='', suffix='', decimals=1, length=40, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if current == total: print()

# ================================================================
# 4. ФОРМАТИРОВАНИЕ ВЫВОДА
# ================================================================

def print_results(res,fout):
    print("\n" + "=" * 70,file=fout)
    print("РЕЗУЛЬТАТЫ ТЕСТА МАННА-ШОЙЕРА-ФЕРТИГА",file=fout)
    print("=" * 70,file=fout)
    print(f"\nРаспределение: {res['distribution']}",file=fout)
    print(f"Объём выборки: {res['n']}",file=fout)
    print(f"Порог gamma: {res['gamma']:.2f}",file=fout)
    #print(f"Параметры: {res['params']}",file=fout)
    print(f"Бета-параметры: n1={res['beta_nu1']}, n2={res['beta_nu2']}",file=fout)
    
    print("\n" + "-" * 50,file=fout)
    print("СТАТИСТИКА КРИТЕРИЯ",file=fout)
    print("-" * 50,file=fout)
    print(f"  M = {res['statistic']:.6f}",file=fout)
    
    print("\n" + "-" * 50,file=fout)
    print("КРИТИЧЕСКИЕ ЗНАЧЕНИЯ (alpha=0.05)",file=fout)
    print("-" * 50,file=fout)
    if not np.isnan(res['critical_beta']):
        print(f"  Бета-аппроксимация ({res['beta_dist']}): {res['critical_beta']:.6f}",file=fout)
    else:
        print(f"  Бета-аппроксимация ({res['beta_dist']}): не определена",file=fout)
    if not np.isnan(res['critical_mc']):
        print(f"  Монте-Карло:        {res['critical_mc']:.6f}",file=fout)
    
    print("\n" + "-" * 50,file=fout)
    print("P-VALUE",file=fout)
    print("-" * 50,file=fout)
    if not np.isnan(res['p_value_beta']):
        print(f"  Бета-аппроксимация: {res['p_value_beta']:.6f}",file=fout)
    if not np.isnan(res['p_value_mc']):
        print(f"  Монте-Карло:        {res['p_value_mc']:.6f}",file=fout)
    
    print("\n" + "-" * 50,file=fout)
    print("ИНТЕРПРЕТАЦИЯ (alpha=0.05)",file=fout)
    print("-" * 50,file=fout)
    if not np.isnan(res['p_value_beta']):
        if res['p_value_beta'] > 0.05:
            print(f"  + H0 НЕ отвергается (p-value = {res['p_value_beta']:.4f} > 0.05)",file=fout)
        else:
            print(f"  - H0 ОТВЕРГАЕТСЯ (p-value = {res['p_value_beta']:.4f} ≤ 0.05)",file=fout)
    else:
        print("  ? Невозможно определить (p-value = nan)",file=fout)


# ================================================================
# 5. MAIN
# ================================================================

def process_mann_test() -> bool:

    inp_dir="Inp"
    out_dir="Out"
    inp_file="mann_test.json"
    out_file="mann_test.out" 
    fout=open(out_dir+"/"+out_file,'w')


    print("=" * 80,file=fout)
    print("ТЕСТ МАННА-ШОЙЕРА-ФЕРТИГА ДЛЯ ПОЛНЫХ ВЫБОРОК",file=fout)
    print("=" * 80,file=fout)
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",file=fout)
    
    # Загрузка конфигурации
    print("\n" + "-" * 40,file=fout)
    print("ЗАГРУЗКА НАСТРОЕК",file=fout)
    print("-" * 40,file=fout)
    
    with open(inp_dir+"/"+inp_file, 'r', encoding='windows-1251') as f:config=json.load(f)
    distribution = config.get('distr_type')
    n_sim = config.get('n_sim')
    use_mc = config.get('use_mc')
    random_seed = config.get('random_seed')
    use_gamma_estimated = config.get('use_gamma_estimated')
    gamma_fixed = config.get('gamma_fixed')
    data=np.array(config.get('data'))
    censored = np.zeros(len(data))
   
    print(f"  Распределение: {distribution}",file=fout)
    print(f"  Итераций Монте-Карло: {n_sim}",file=fout)
    print(f"  Использовать MC: {use_mc}",file=fout)
    print(f"  Random seed: {random_seed}",file=fout)
    print(f"  Использовать оцененный gamma: {use_gamma_estimated}",file=fout)
    if not use_gamma_estimated:   print(f"  Фиксированный gamma: {gamma_fixed}",file=fout)
    
    print(f"  Размер выборки: {len(data)}",file=fout)
    print(f"  Диапазон: [{np.min(data):.2e}, {np.max(data):.2e}]",file=fout)
    if use_gamma_estimated:
        print("\n" + "-" * 40,file=fout)
        print("ПОИСК ОПТИМАЛЬНОГО gamma",file=fout)
        print("-" * 40,file=fout)
        if distribution == 'BS':
            fit_result = find_optimal_gamma_bs(data, censored, 0.0, 0.99, 1e-6, 50)
            gamma = fit_result['optimal_gamma']
        elif distribution == 'Weibull':
            fit_result = find_optimal_gamma_w(data, censored, 0.0, 0.99, 1e-6, 50)
            gamma = fit_result['optimal_gamma']
        elif distribution == 'Normal':
            # Для логнормального γ = 0 (обычно)
            gamma = 0.0
            print(f"  Для логнормального распределения γ = 0.00",file=fout)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        if distribution != 'Normal':
            print(f"  Оптимальный gamma = {gamma:.2f}",file=fout)
    else:
        gamma = gamma_fixed
        print(f"\n  Используется фиксированный gamma = {gamma:.2f}",file=fout)
    
    # Вывод
    
    print("\n" + "-" * 40,file=fout)
    print("ВЫПОЛНЕНИЕ РАСЧЁТОВ",file=fout)
    print("-" * 40,file=fout)
    
    result = run_mann_test(data, distribution, gamma, n_sim, use_mc, random_seed)
    print_results(result,fout)
    
    fout.close()
    return True

#=======================================================
if __name__ == "__main__":
    process_mann_test()
