import numpy as np
import json
from datetime import datetime
from scipy import stats
from scipy.stats import shapiro
import warnings
warnings.filterwarnings('ignore')
# ===================== Импорт из estimation.py ====================
import estimation
from estimation import (
    fit_bs_fixed_loc,
    find_optimal_gamma_bs
)


"""
================================================================================
ТЕСТ ШАПИРО-УИЛКА ДЛЯ ПОЛНЫХ ВЫБОРОК
================================================================================
Критерий Шапиро-Уилка для проверки нормальности.
Поддерживает:
    - BS (преобразование к нормальности)
    - Normal (непосредственно)
Для BS выполняется преобразование:
    Z = (1/α) * (√(X/β) - √(β/X))
Тип распределения задаётся в shapiro_test.json (distr_type)
"""

# Критические значения W для alpha = 0.05, 0.1, 0.01 (n=3..50)
wcr5 = [0.767, 0.748, 0.762, 0.788, 0.803, 0.818, 0.829, 0.842, 0.850, 0.859,
        0.866, 0.874, 0.881, 0.887, 0.892, 0.897, 0.901, 0.905, 0.908, 0.911,
        0.914, 0.916, 0.918, 0.920, 0.923, 0.924, 0.926, 0.927, 0.929, 0.930,
        0.931, 0.933, 0.934, 0.935, 0.936, 0.938, 0.939, 0.940, 0.941, 0.942,
        0.943, 0.944, 0.945, 0.945, 0.946, 0.947, 0.947, 0.947]

wcr10 = [0.789, 0.792, 0.806, 0.826, 0.838, 0.851, 0.859, 0.869, 0.876, 0.883,
         0.889, 0.895, 0.901, 0.906, 0.910, 0.914, 0.917, 0.920, 0.923, 0.926,
         0.928, 0.930, 0.931, 0.933, 0.935, 0.936, 0.937, 0.939, 0.940, 0.941,
         0.942, 0.943, 0.944, 0.945, 0.946, 0.947, 0.948, 0.949, 0.950, 0.951,
         0.951, 0.952, 0.953, 0.953, 0.954, 0.954, 0.955, 0.955]

wcr01 = [0.753, 0.687, 0.686, 0.713, 0.730, 0.749, 0.764, 0.781, 0.792, 0.805,
         0.814, 0.825, 0.835, 0.844, 0.851, 0.858, 0.863, 0.868, 0.873, 0.878,
         0.881, 0.884, 0.888, 0.891, 0.894, 0.896, 0.898, 0.900, 0.902, 0.904,
         0.906, 0.908, 0.910, 0.912, 0.914, 0.916, 0.917, 0.919, 0.920, 0.922,
         0.923, 0.924, 0.926, 0.927, 0.928, 0.929, 0.929, 0.930]

#============================================================

def shapiro_critical_approx(n, alpha=0.05):
    if n < 3:   return np.nan
    if n > 50:
        n_idx = 47
    else:
        n_idx = n - 3
    
    if alpha == 0.01:
        crit = wcr01[n_idx]
    elif alpha == 0.05:
        crit = wcr5[n_idx]
    elif alpha == 0.10:
        crit = wcr10[n_idx]
    else:
        crit = wcr5[n_idx]
    return crit

# ================================================================
# 1. ПРЕОБРАЗОВАНИЕ BS К НОРМАЛЬНОСТИ
# ================================================================

def bs_to_normal(data, gamma):
    n = len(data)
    censored = np.zeros(n)
    data_adj = data - gamma
    # Оценка параметров BS
    fit_result = fit_bs_fixed_loc(data, censored, gamma)
    alpha_hat = fit_result['alpha']
    beta_hat = fit_result['beta']
    # Преобразование
    z = (np.sqrt(data_adj / beta_hat) - np.sqrt(beta_hat / data_adj)) / alpha_hat
    return z, alpha_hat, beta_hat, fit_result


# ================================================================
# 2. ТЕСТ ШАПИРО-УИЛКА
# ================================================================

def shapiro_test_full(x, alpha=0.05):

    n = len(x)
    w_stat, p_value = shapiro(x)
    w_crit = shapiro_critical_approx(n, alpha)
    return {
        'statistic': w_stat,
        'p_value': p_value,
        'critical_approx': w_crit,
        'n': n
    }


# ================================================================
# 3. ОСНОВНАЯ ФУНКЦИЯ ЗАПУСКА
# ================================================================

def prepare_data_shapiro(data, distribution, gamma):

    if distribution == 'BS':
        z, alpha_hat, beta_hat, fit_result = bs_to_normal(data, gamma)
        params_str = f"α={alpha_hat:.6f}, β={beta_hat:.2f}, γ={gamma:.2f}"
        return z, params_str, fit_result
    elif distribution == 'Normal':
        z = np.log10(data.copy())
        params_str = f"нормальное, γ={gamma:.2f}"
        fit_result = {'distribution': 'Normal', 'gamma': gamma}
        return z, params_str, fit_result
    else:
        raise ValueError(f"Тест Шапиро-Уилка не поддерживается для {distribution}")

#=======================================================================

def run_shapiro_test(data, distribution, gamma, alpha):
    z, params_str, fit_result = prepare_data_shapiro(data, distribution, gamma)
    result = shapiro_test_full(z, alpha)
    result['distribution'] = distribution
    result['params'] = params_str
    result['gamma'] = gamma
    result['n'] = len(data)
    result['fit_result'] = fit_result
    return result

#========================================================

def print_results_shapiro(res,fout):
    print("\n" + "=" * 70,file=fout)
    print("РЕЗУЛЬТАТЫ ТЕСТА ШАПИРО-УИЛКА",file=fout)
    print("=" * 70,file=fout)
    print(f"\nРаспределение: {res['distribution']}",file=fout)
    print(f"Объём выборки: {res['n']}",file=fout)
    print(f"Порог gamma: {res['gamma']:.2f}",file=fout)
    
    print("\n" + "-" * 50,file=fout)
    print("СТАТИСТИКА КРИТЕРИЯ",file=fout)
    print("-" * 50,file=fout)
    print(f"  W = {res['statistic']:.6f}",file=fout)
    
    print("\n" + "-" * 50,file=fout)
    print("P-VALUE",file=fout)
    print("-" * 50,file=fout)
    print(f"  p-value = {res['p_value']:.6f}",file=fout)
    
    print("\n" + "-" * 50,file=fout)
    print("КРИТИЧЕСКОЕ ЗНАЧЕНИЕ (alpha=0.05)",file=fout)
    print("-" * 50,file=fout)
    if not np.isnan(res['critical_approx']):
        print(f"  по таблице: {res['critical_approx']:.6f}",file=fout)
    else:
        print(f"  Таблица определена",file=fout)

    print("\n" + "-" * 50,file=fout)
    print("ИНТЕРПРЕТАЦИЯ (alpha=0.05)",file=fout)
    print("-" * 50,file=fout)
    if not np.isnan(res['p_value']):
        if res['p_value'] > 0.05:
            print(f"   H0+ НЕ отвергается (p-value = {res['p_value']:.4f} > 0.05)",file=fout)
        else:
            print(f"   H0- ОТВЕРГАЕТСЯ (p-value = {res['p_value']:.4f} ≤ 0.05)",file=fout)
    else:
        print("  ? Невозможно определить (p-value = nan)",file=fout)
    
    # Дополнительная проверка по критическому значению
    if not np.isnan(res['critical_approx']):
        if res['statistic'] > res['critical_approx']:
            print(f"\n  Дополнительно: W = {res['statistic']:.4f} > W_crit = {res['critical_approx']:.4f}",file=fout)
            print(f"   H0+ НЕ отвергается",file=fout)
        else:
            print(f"\n  Дополнительно: W = {res['statistic']:.4f} ≤ W_crit = {res['critical_approx']:.4f}",file=fout)
            print(f"   H0- ОТВЕРГАЕТСЯ",file=fout)


# ================================================================
# 5. MAIN
# ================================================================

def process_shapiro_test() -> bool:

    inp_dir="Inp"
    out_dir="Out"
    inp_file="shapiro_test.json"
    out_file="shapiro_test.out" 
    fout=open(out_dir+"/"+out_file,'w')

    print("=" * 80,file=fout)
    print("ТЕСТ ШАПИРО-УИЛКА ДЛЯ ПОЛНЫХ ВЫБОРОК",file=fout)
    print("=" * 80,file=fout)
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",file=fout)
    
    # Загрузка конфигурации
    print("\n" + "-" * 40,file=fout)
    print("ЗАГРУЗКА НАСТРОЕК",file=fout)
    print("-" * 40,file=fout)
    
    with open(inp_dir+"/"+inp_file, 'r', encoding='windows-1251') as f:config=json.load(f)

    distribution = config.get('distr_type')
    use_gamma_estimated = config.get('use_gamma_estimated')
    gamma_fixed = config.get('gamma_fixed')
    alpha = config.get('alpha')
    data=np.array(config.get('data'))
    censored = np.zeros(len(data))
    
    print(f"  Распределение: {distribution}",file=fout)
    print(f"  Уровень значимости: {alpha}",file=fout)
    print(f"  Использовать оцененный gamma: {use_gamma_estimated}",file=fout)
    if not use_gamma_estimated:   print(f"  Фиксированный gamma: {gamma_fixed}",file=fout)
    
    # Проверка: Шапиро-Уилк работает только для BS и Normal
    if distribution not in ['BS', 'Normal']:
        print(f"\n  ВНИМАНИЕ: Тест Шапиро-Уилка не поддерживается для {distribution}",file=fout)
        print("  Поддерживаемые: BS, Normal",file=fout)
        fout.close()
        return False
        
    print(f"  Размер выборки: {len(data)}",file=fout)
    print(f"  Диапазон: [{np.min(data):.2e}, {np.max(data):.2e}]",file=fout)
    
    if use_gamma_estimated:
        print("\n" + "-" * 40,file=fout)
        print("ПОИСК ОПТИМАЛЬНОГО gamma",file=fout)
        print("-" * 40,file=fout)
        if distribution == 'BS':
            fit_result = find_optimal_gamma_bs(data, censored, 0.0, 0.99, 1e-6, 50)
            gamma=fit_result['optimal_gamma']
            print(f"  gamma={gamma:.2f}",file=fout)
        elif distribution == 'Normal':
            gamma=0.0
            print(f"  gamma={gamma:.2f}",file=fout)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    else:
        gamma = gamma_fixed
        print(f"\n  Используется фиксированный gamma={gamma:.2f}",file=fout)
    
    # Вывод
    
    print("\n" + "-" * 40,file=fout)
    print("ВЫПОЛНЕНИЕ РАСЧЕТОВ",file=fout)
    print("-" * 40,file=fout)
    result = run_shapiro_test(data, distribution, gamma, alpha)
    print_results_shapiro(result,fout)
    
    fout.close()
    return True

#==================================================

if __name__ == "__main__":
    process_shapiro_test()

