"""
ДВУХЭТАПНАЯ ОПТИМИЗАЦИЯ ПЛАНИРОВАНИЯ УСТАЛОСТНЫХ ИСПЫТАНИЙ
Этап 1: Теоретическая оптимизация (на основе информационной матрицы Фишера)
Этап 2: Уточнение методом Монте-Карло
Поддержка произвольного числа уровней нагружения (3, 4, 5 и более)
"""

import numpy as np
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from scipy.stats import t
import time
import hashlib
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MaterialParams:
    sigma_inf: float
    C: float
    m: float
    a: float
    alpha: float
    lgN_levels: List[float]
    N0_list: List[float]
    C1: float
    C3: float
    f: float
    eps: float = 0.35
    gamma: float = 0.1
    t_quantile: float = 2.0
    n_simulations: int = 300
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MaterialParams':
        return cls(
            sigma_inf=data['sigma_inf'],
            C=data['C'],
            m=data['m'],
            a=data['a'],
            alpha=data['alpha'],
            lgN_levels=data['lgN_levels'],
            N0_list=data['N0_list'],
            C1=data['C1'],
            C3=data['C3'],
            f=data['f'],
            eps=data.get('eps', 0.35),
            gamma=data.get('gamma', 0.1),
            t_quantile=data.get('t_quantile', 2.0),
            n_simulations=data.get('n_simulations', 300)
        )
    
    def sigma_from_N(self, N: float) -> float:
        return self.sigma_inf + self.C * (N) ** (-self.m)
    
    def get_delta_target(self) -> float:
        return self.eps * self.gamma / self.t_quantile


@dataclass
class Plan:
    nu: List[float]
    n: int
    delta: float
    cost: float
    description: str = ""
    n_distribution: List[int] = None


# ====================== ТЕОРЕТИЧЕСКАЯ ОЦЕНКА ======================

def theoretical_std_sigma_R(params: MaterialParams, n: int, nu: List[float], N0: float) -> float:
    nu = np.array(nu)
    n_i = n * nu
    sigma_levels = np.array([params.sigma_from_N(10 ** lgN) for lgN in params.lgN_levels])
    lgN_levels = np.array(params.lgN_levels)
    variances = (params.a * (lgN_levels ** params.alpha)) ** 2
    weights = n_i / variances
    sigma_adj = sigma_levels - params.sigma_inf
    ln10 = np.log(10)
    sigma_adj = np.maximum(sigma_adj, 1e-6)
    d_dsigma_inf = (1.0 / params.m) * (1.0 / (ln10 * sigma_adj))
    d_dC = (1.0 / params.m) * (1.0 / (ln10 * params.C)) * np.ones_like(sigma_levels)
    d_dm = -(1.0 / params.m ** 2) * (np.log10(params.C) - np.log10(sigma_adj))
    J = np.column_stack([d_dsigma_inf, d_dC, d_dm])
    Fisher = J.T @ np.diag(weights) @ J
    if np.linalg.cond(Fisher) > 1e12:
        return 0.0
    
    try:
        Cov = np.linalg.inv(Fisher)
    except np.linalg.LinAlgError:
        return 0.0
    
    term = N0 ** (-params.m)
    dR_dsigma_inf = 1.0
    dR_dC = term
    dR_dm = -params.C * term * np.log(N0)
    dR = np.array([dR_dsigma_inf, dR_dC, dR_dm])
    
    var_sigma_R = dR @ Cov @ dR
    return np.sqrt(max(var_sigma_R, 0.0))


def theoretical_delta(params: MaterialParams, n: int, nu: List[float], N0: float) -> float:
    true_val = params.sigma_from_N(N0)
    std_theo = theoretical_std_sigma_R(params, n, nu, N0)
    if true_val <= 0 or std_theo <= 0:
        return 0.25
    return std_theo / true_val


def theoretical_min_n(params: MaterialParams, nu: List[float], N0: float,
                      n_min: int = 4, n_max: int = 300) -> Tuple[int, float]:
    delta_target = params.get_delta_target()
    delta_max = theoretical_delta(params, n_max, nu, N0)
    if delta_max > delta_target:
        return n_max, delta_max
    
    left, right = n_min, n_max
    best_n = n_max
    
    while left <= right:
        mid = (left + right) // 2
        delta = theoretical_delta(params, mid, nu, N0)
        if delta <= delta_target:
            best_n = mid
            right = mid - 1
        else:
            left = mid + 1
    return best_n, theoretical_delta(params, best_n, nu, N0)


def calculate_cost(params: MaterialParams, n: int, nu: List[float]) -> float:
    nu = np.array(nu)
    n_i = np.round(n * nu).astype(int)
    # Корректировка суммы и гарантия минимум 1 образец на уровень
    if np.sum(n_i) != n:
        # Находим уровень с максимальным n_i и добавляем недостающие
        idx = np.argmax(n_i)
        n_i[idx] += n - np.sum(n_i)
    # Гарантируем минимум 1 образец на уровень (если nu[i] > 0)
    for i in range(len(n_i)):
        if nu[i] > 0 and n_i[i] == 0:
            n_i[i] = 1
            # Уменьшаем самый большой уровень
            idx = np.argmax(n_i)
            n_i[idx] -= 1
    
    if np.any(n_i < 0):
        return np.inf
    
    total_cycles = 0
    for i, lgN in enumerate(params.lgN_levels):
        if n_i[i] > 0:
            total_cycles += n_i[i] * (10 ** lgN)
    time_hours = total_cycles / (params.f * 3600)
    cost = n + (params.C3 / params.C1) * time_hours
    return cost


# ====================== ГЕНЕРАЦИЯ ВАРИАНТОВ РАСПРЕДЕЛЕНИЯ ======================

def generate_nu_variants(params: MaterialParams) -> List[Tuple[float, List[float], str]]:
    """
    Генерирует варианты распределений ν для произвольного числа уровней
    """
    n_levels = len(params.lgN_levels)
    lgN_levels = np.array(params.lgN_levels)
    lgN_min = lgN_levels[0]
    lgN_max = lgN_levels[-1]
    eps = 0.01
    
    variants = []
    
    # ====================== 1. РАВНОМЕРНОЕ ======================
    nu_uniform = np.ones(n_levels) / n_levels
    variants.append((0.0, nu_uniform.tolist(), "Равномерное"))
    
    # ====================== 2. СДВИГ ВПРАВО (степенной закон) ======================
    p_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
    
    for p in p_values:
        lgN_norm = (lgN_levels - lgN_min) / (lgN_max - lgN_min)
        weights = (lgN_norm + eps) ** p
        nu = weights / np.sum(weights)
        desc = f"Сдвиг вправо p={p:.1f}"
        variants.append((p, nu.tolist(), desc))
    
    # ====================== 3. СДВИГ ВЛЕВО (степенной закон) ======================
    for p in p_values:
        lgN_norm = (lgN_max - lgN_levels) / (lgN_max - lgN_min)
        weights = (lgN_norm + eps) ** p
        nu = weights / np.sum(weights)
        desc = f"Сдвиг влево p={p:.1f}"
        variants.append((-p, nu.tolist(), desc))
    
    # ====================== 4. СИММЕТРИЧНЫЕ ВАРИАНТЫ (для 3,4,5 уровней) ======================
    symmetric_configs = generate_symmetric_configs(n_levels)
    for nu_vals, desc in symmetric_configs:
        nu = np.array(nu_vals) / np.sum(nu_vals)
        variants.append((0.0, nu.tolist(), desc))
    
    # ====================== 5. ЭКСТРЕМАЛЬНЫЕ ОДНОСТОРОННИЕ ======================
    extreme_configs = generate_extreme_configs(n_levels)
    for nu_vals, desc in extreme_configs:
        nu = np.array(nu_vals) / np.sum(nu_vals)
        variants.append((0.0, nu.tolist(), desc))
    
    return variants


def generate_symmetric_configs(n_levels: int) -> List[Tuple[List[float], str]]:
    """
    Генерирует симметричные конфигурации для произвольного числа уровней
    """
    configs = []
    
    if n_levels == 3:
        configs = [
            ([0.4, 0.6, 0.4], "Симметричный (0.4,0.6,0.4)"),
            ([0.3, 0.7, 0.3], "Симметричный (0.3,0.7,0.3)"),
            ([0.2, 0.8, 0.2], "Симметричный (0.2,0.8,0.2)"),
            ([0.5, 0.5, 0.5], "Симметричный (0.5,0.5,0.5)"),
            ([0.6, 0.4, 0.6], "Симметричный (0.6,0.4,0.6)"),
        ]
    elif n_levels == 4:
        configs = [
            ([0.4, 0.1, 0.1, 0.4], "Симметричный (0.4,0.1,0.1,0.4)"),
            ([0.35, 0.15, 0.15, 0.35], "Симметричный (0.35,0.15,0.15,0.35)"),
            ([0.3, 0.2, 0.2, 0.3], "Симметричный (0.3,0.2,0.2,0.3)"),
            ([0.25, 0.25, 0.25, 0.25], "Симметричный (0.25,0.25,0.25,0.25)"),
            ([0.2, 0.3, 0.3, 0.2], "Симметричный (0.2,0.3,0.3,0.2)"),
            ([0.15, 0.35, 0.35, 0.15], "Симметричный (0.15,0.35,0.35,0.15)"),
            ([0.1, 0.4, 0.4, 0.1], "Симметричный (0.1,0.4,0.4,0.1)"),
            ([0.45, 0.05, 0.05, 0.45], "Симметричный (0.45,0.05,0.05,0.45)"),
            ([0.48, 0.02, 0.02, 0.48], "Симметричный (0.48,0.02,0.02,0.48)"),
        ]
    elif n_levels == 5:
        configs = [
            ([0.3, 0.1, 0.1, 0.1, 0.3], "Симметричный (0.3,0.1,0.1,0.1,0.3)"),
            ([0.25, 0.15, 0.1, 0.15, 0.25], "Симметричный (0.25,0.15,0.1,0.15,0.25)"),
            ([0.2, 0.2, 0.2, 0.2, 0.2], "Симметричный (0.2,0.2,0.2,0.2,0.2)"),
            ([0.15, 0.25, 0.2, 0.25, 0.15], "Симметричный (0.15,0.25,0.2,0.25,0.15)"),
            ([0.1, 0.3, 0.2, 0.3, 0.1], "Симметричный (0.1,0.3,0.2,0.3,0.1)"),
            ([0.05, 0.35, 0.2, 0.35, 0.05], "Симметричный (0.05,0.35,0.2,0.35,0.05)"),
        ]
    else:
        # Для большего числа уровней — равномерное распределение по краям
        mid = n_levels // 2
        left = [0.1] * (mid + 1)
        right = [0.1] * (n_levels - mid - 1)
        configs.append((left + right, f"Симметричный (края 0.1)"))
        configs.append(([0.2] * n_levels, f"Равномерный (0.2)"))
    
    return configs


def generate_extreme_configs(n_levels: int) -> List[Tuple[List[float], str]]:
    """
    Генерирует экстремальные односторонние конфигурации
    """
    configs = []
    
    if n_levels == 3:
        configs = [
            ([0.1, 0.2, 0.7], "Экстремальный вправо (0.1,0.2,0.7)"),
            ([0.7, 0.2, 0.1], "Экстремальный влево (0.7,0.2,0.1)"),
            ([0.05, 0.15, 0.8], "Экстремальный вправо (0.05,0.15,0.8)"),
            ([0.8, 0.15, 0.05], "Экстремальный влево (0.8,0.15,0.05)"),
        ]
    elif n_levels == 4:
        configs = [
            ([0.1, 0.1, 0.3, 0.5], "Экстремальный вправо (0.1,0.1,0.3,0.5)"),
            ([0.05, 0.1, 0.25, 0.6], "Экстремальный вправо (0.05,0.1,0.25,0.6)"),
            ([0.02, 0.08, 0.2, 0.7], "Экстремальный вправо (0.02,0.08,0.2,0.7)"),
            ([0.5, 0.3, 0.1, 0.1], "Экстремальный влево (0.5,0.3,0.1,0.1)"),
            ([0.6, 0.25, 0.1, 0.05], "Экстремальный влево (0.6,0.25,0.1,0.05)"),
            ([0.7, 0.2, 0.08, 0.02], "Экстремальный влево (0.7,0.2,0.08,0.02)"),
        ]
    elif n_levels == 5:
        configs = [
            ([0.05, 0.05, 0.1, 0.3, 0.5], "Экстремальный вправо (0.05,0.05,0.1,0.3,0.5)"),
            ([0.5, 0.3, 0.1, 0.05, 0.05], "Экстремальный влево (0.5,0.3,0.1,0.05,0.05)"),
        ]
    
    return configs


# ====================== ЭТАП 1: ТЕОРЕТИЧЕСКАЯ ОПТИМИЗАЦИЯ ======================

def theoretical_optimization(params: MaterialParams, N0: float,fout) -> Plan:

    #Теоретическая оптимизация с приоритетом точности

    print(f"\n  Теоретическая оптимизация для N0={N0:.0e}...",file=fout)
    variants = generate_nu_variants(params)
    results = []
    n_levels = len(params.lgN_levels)
    header_nu = "ν1...νn"
    #print(f"\n {'Вариант':<40} {'n':<4} {'delta':<10} {'cost':<8} {header_nu}",file=fout)
    #print(f"    {'-'*40} {'-'*4} {'-'*10} {'-'*8} {'-'*50}",file=fout)
    delta_target = params.get_delta_target()
    for beta, nu, desc in variants:
        n_opt, delta_opt = theoretical_min_n(params, nu, N0)
        cost = calculate_cost(params, n_opt, nu)
        n_dist = np.round(n_opt * np.array(nu)).astype(int)
        # Гарантия минимум 1 образец на уровень (если nu[i] > 0)
        for i in range(len(n_dist)):
            if nu[i] > 0 and n_dist[i] == 0:
                n_dist[i] = 1
                idx = np.argmax(n_dist)
                n_dist[idx] -= 1
        if np.sum(n_dist) != n_opt:
            idx = np.argmax(n_dist)
            n_dist[idx] += n_opt - np.sum(n_dist)
        results.append({
            'beta': beta,
            'nu': nu,
            'n': n_opt,
            'n_dist': n_dist.tolist(),
            'delta': delta_opt,
            'cost': cost,
            'description': desc
        })
        status = "+" if delta_opt <= delta_target else "-"
        nu_str = ", ".join([f"{x:.3f}" for x in nu])
        print(f"    {desc:<40} {n_opt:>3}   {delta_opt:.5f}   {cost:>6.1f}   [{nu_str}] {status}",file=fout)
    # Выбор лучшего плана с приоритетом точности
    valid_plans = [r for r in results if r['delta'] <= delta_target]
    if valid_plans:
        best = min(valid_plans, key=lambda x: x['cost'])
        print(f"\n    Лучший теоретический план (с достигнутой точностью):",file=fout)
    else:
        best = min(results, key=lambda x: x['delta'])
        print(f"\n    Лучший теоретический план (точность не достигнута, минимальная delta):",file=fout)
    print(f"      {best['description']}",file=fout)
    print(f"      nu = {[round(x, 4) for x in best['nu']]}",file=fout)
    print(f"      n = {best['n']} - распределение: {best['n_dist']}",file=fout)
    print(f"      delta = {best['delta']:.5f} (цель<={delta_target:.5f})",file=fout)
    print(f"      cost = {best['cost']:.1f}",file=fout)
    return Plan(
        nu=best['nu'],
        n=best['n'],
        delta=best['delta'],
        cost=best['cost'],
        description=best['description'],
        n_distribution=best['n_dist']
    )


# ====================== ЭТАП 2: МОНТЕ-КАРЛО УТОЧНЕНИЕ ======================

class MonteCarloEstimator:
    def __init__(self, params: MaterialParams, N0: float, n_simulations: int):
        self.params = params
        self.N0 = N0
        self.n_simulations = n_simulations
        self.cache = {}
        self.strength_cache = {}
        self.hits = 0
        self.misses = 0
        
    def sn_curve(self, sigma: np.ndarray, sigma_inf: float, C: float, m: float) -> np.ndarray:
        sigma_adj = sigma - sigma_inf
        sigma_adj = np.maximum(sigma_adj, 1e-6)
        return (1.0 / m) * (np.log10(C) - np.log10(sigma_adj))
    
    def get_cache_key(self, n: int, nu: np.ndarray, seed: int) -> str:
        key_str = f"{n}_{tuple(np.round(nu, 4))}_{seed}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def generate_experiment(self, n: int, nu: np.ndarray, seed: int) -> np.ndarray:
        np.random.seed(seed)
        nu = np.array(nu)
        n_i = np.round(n * nu).astype(int)
        
        # Корректировка суммы и гарантия минимум 1 образец
        if np.sum(n_i) != n:
            idx = np.argmax(n_i)
            n_i[idx] += n - np.sum(n_i)
        
        for i in range(len(n_i)):
            if nu[i] > 0 and n_i[i] == 0:
                n_i[i] = 1
                idx = np.argmax(n_i)
                n_i[idx] -= 1
        
        if np.any(n_i < 0):
            return np.array([])
        
        data = []
        for i, lgN_mean in enumerate(self.params.lgN_levels):
            if n_i[i] > 0:
                sigma_lgN = self.params.a * lgN_mean ** self.params.alpha
                lgN_sample = np.random.normal(lgN_mean, sigma_lgN, n_i[i])
                sigma_level = self.params.sigma_from_N(10 ** lgN_mean)
                for lgN in lgN_sample:
                    data.append([sigma_level, lgN])
        return np.array(data)
    
    def estimate_strength(self, data: np.ndarray) -> Optional[float]:
        if len(data) < 4:
            return None
        sigma = data[:, 0]
        lgN = data[:, 1]
        
        p0 = [self.params.sigma_inf, self.params.C, self.params.m]
        bounds_lower = [200, 400, 0.08]
        bounds_upper = [300, 1600, 0.20]
        
        try:
            popt, _ = curve_fit(self.sn_curve, sigma, lgN, p0=p0,
                               bounds=(bounds_lower, bounds_upper),
                               maxfev=5000)
            sigma_inf_est, C_est, m_est = popt
            return sigma_inf_est + C_est * (self.N0) ** (-m_est)
        except:
            return None
    
    def compute_delta(self, n: int, nu: np.ndarray, use_cache: bool = True) -> float:
        cache_key = (n, tuple(np.round(nu, 4)))
        
        if use_cache and cache_key in self.cache:
            self.hits += 1
            return self.cache[cache_key]
        
        self.misses += 1
        estimates = []
        
        for seed in range(self.n_simulations):
            strength_key = self.get_cache_key(n, nu, seed)
            if use_cache and strength_key in self.strength_cache:
                est = self.strength_cache[strength_key]
            else:
                data = self.generate_experiment(n, nu, seed + 10000)
                est = self.estimate_strength(data)
                if use_cache and est is not None:
                    self.strength_cache[strength_key] = est
            
            if est is not None:
                estimates.append(est)
        
        if len(estimates) < 20:
            delta = 0.25
        else:
            mean_est = np.mean(estimates)
            if mean_est <= 0:
                delta = 0.25
            else:
                delta = np.std(estimates) / mean_est
        
        if use_cache:
            self.cache[cache_key] = delta
        
        return delta



def monte_carlo_optimization(params: MaterialParams, theoretical_plan: Plan, N0: float,fout) -> Plan:
    print(f"\n  Этап 2: Уточнение методом Монте-Карло (симуляций={params.n_simulations})...",file=fout)
    
    nu = theoretical_plan.nu
    mc = MonteCarloEstimator(params, N0, params.n_simulations)
    delta_target = params.get_delta_target()
    
    n_current = theoretical_plan.n
    delta_current = mc.compute_delta(n_current, nu)
    print(f"    Теоретический план: n={n_current}, delta={delta_current:.5f}",file=fout)
    
    # Границы поиска (с запасом)
    n_min = 4
    n_max = max(n_current * 2, 500)  # Максимум 500 образцов
    
    # Проверяем, достижима ли точность при n_max
    delta_max = mc.compute_delta(n_max, nu)
    
    if delta_max > delta_target:
        print(f"     Предупреждение: даже при n={n_max} точность не достигнута (delta={delta_max:.5f} > {delta_target:.5f})",file=fout)
        best_n = n_max
        best_delta = delta_max
    else:
        print(f"    Бинарный поиск оптимального n в диапазоне [{n_min}, {n_max}]...",file=fout)
        
        # Бинарный поиск минимального n, при котором delta ≤ delta_target
        left, right = n_min, n_max
        best_n = n_max
        best_delta = delta_max
        
        iteration = 0
        while left <= right:
            mid = (left + right) // 2
            delta_mid = mc.compute_delta(mid, nu)
            iteration += 1
            print(f"      Итерация {iteration}: n={mid}, delta={delta_mid:.5f}",file=fout)
            
            if delta_mid <= delta_target:
                # Точность достигнута, пробуем уменьшить n
                best_n = mid
                best_delta = delta_mid
                right = mid - 1
            else:
                # Точность не достигнута, увеличиваем n
                left = mid + 1
        
        n_current = best_n
        delta_current = best_delta
        
        # Дополнительная проверка: возможно, можно ещё немного уменьшить n
        # (уточнение в окрестности найденного значения)
        for n_candidate in [n_current - 1, n_current - 2]:
            if n_candidate >= n_min:
                delta_candidate = mc.compute_delta(n_candidate, nu)
                if delta_candidate <= delta_target:
                    n_current = n_candidate
                    delta_current = delta_candidate
                    print(f"    Уточнение: удалось уменьшить до n={n_current}, delta={delta_current:.5f}",file=fout)
    
    # Формируем распределение образцов
    n_dist = np.round(n_current * np.array(nu)).astype(int)
    
    # Гарантия минимум 1 образец на уровень (если nu[i] > 0)
    for i in range(len(n_dist)):
        if nu[i] > 0 and n_dist[i] == 0:
            n_dist[i] = 1
            # Уменьшаем самый большой уровень, если сумма превышает n_current
            if np.sum(n_dist) > n_current:
                idx = np.argmax(n_dist)
                n_dist[idx] -= 1
    
    # Корректировка суммы до n_current
    if np.sum(n_dist) != n_current:
        idx = np.argmax(n_dist)
        n_dist[idx] += n_current - np.sum(n_dist)
    
    # Проверка на отрицательные значения
    if np.any(n_dist < 0):
        print(f"     Ошибка: отрицательные значения в распределении, исправляем...",file=fout)
        n_dist = np.maximum(n_dist, 0)
        idx = np.argmax(n_dist)
        n_dist[idx] += n_current - np.sum(n_dist)
    
    cost = calculate_cost(params, n_current, nu)
    
    print(f"\n     Финальный план (Монте-Карло):",file=fout)
    print(f"      nu = {[round(x, 4) for x in nu]}",file=fout)
    print(f"      n = {n_current} - распределение: {n_dist.tolist()}",file=fout)
    print(f"      delta = {delta_current:.5f} (цель <= {delta_target:.5f})",file=fout)
    print(f"      cost = {cost:.1f}",file=fout)
    
    return Plan(
        nu=nu,
        n=n_current,
        delta=delta_current,
        cost=cost,
        description=theoretical_plan.description,
        n_distribution=n_dist.tolist()
    )


# ====================== MAIN ======================

def process_fatigue_optimizer() -> bool:

    inp_dir="Inp"
    out_dir="Out"
    inp_file="fatigue_optimizer.json"
    out_file="fatigue_optimizer.out" 
    fout=open(out_dir+"/"+out_file,'w')
    
    print("=" * 70,file=fout)
    print("ДВУХЭТАПНАЯ ОПТИМИЗАЦИЯ ПЛАНИРОВАНИЯ УСТАЛОСТНЫХ ИСПЫТАНИЙ",file=fout)
    print("Этап 1: Теоретическая оптимизация (на основе информационной матрицы)",file=fout)
    print("Этап 2: Уточнение методом Монте-Карло",file=fout)
    print("Поддержка произвольного числа уровней нагружения",file=fout)
    print("=" * 70,file=fout)
    
    with open(inp_dir+"/"+inp_file, 'r', encoding='utf-8') as f:params_dict = json.load(f)
    
    params = MaterialParams.from_dict(params_dict)
    
    print(f"\nПараметры задачи:",file=fout)
    print(f"  s_inf = {params.sigma_inf} МПа",file=fout)
    print(f"  C = {params.C}, m = {params.m}",file=fout)
    print(f"  a = {params.a}, alpha = {params.alpha}",file=fout)
    print(f"  Уровни lgN ({len(params.lgN_levels)} шт): {params.lgN_levels}",file=fout)
    print(f"  C3/C1 = {params.C3/params.C1:.3f}",file=fout)
    print(f"  eps = {params.eps}, gamma = {params.gamma}, t = {params.t_quantile}",file=fout)
    print(f"  delta_target = {params.get_delta_target():.5f}",file=fout)
    print(f"  f = {params.f} Гц",file=fout)
    print(f"  n_simulations = {params.n_simulations}",file=fout)
    
    results = {}
    
    for N0 in params.N0_list:
        print(f"\n{'=' * 70}",file=fout)
        print(f"ОПТИМИЗАЦИЯ ДЛЯ N0 = {N0:.0e}",file=fout)
        print(f"{'=' * 70}",file=fout)
        
        theoretical_plan = theoretical_optimization(params, N0,fout)
        final_plan = monte_carlo_optimization(params, theoretical_plan, N0,fout)
        
        results[f"N0_{N0:.0e}"] = {
            'theoretical': {
                'nu': theoretical_plan.nu,
                'n': theoretical_plan.n,
                'n_distribution': theoretical_plan.n_distribution,
                'delta': theoretical_plan.delta,
                'cost': theoretical_plan.cost,
                'description': theoretical_plan.description
            },
            'monte_carlo': {
                'nu': final_plan.nu,
                'n': final_plan.n,
                'n_distribution': final_plan.n_distribution,
                'delta': final_plan.delta,
                'cost': final_plan.cost
            }
        }
    
    print("\n" + "=" * 70,file=fout)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ",file=fout)
    print("=" * 70,file=fout)
    print(f"\n{'N0':<12} {'Этап':<15} {'n':<6} {'delta':<10} {'cost':<10} {'Распределение (n1...nk)'}",file=fout)
    print("-" * 80,file=fout)
    
    for key, result in results.items():
        theo = result['theoretical']
        mc = result['monte_carlo']
        
        theo_dist = ", ".join([str(x) for x in theo['n_distribution']]) if theo['n_distribution'] else "—"
        mc_dist = ", ".join([str(x) for x in mc['n_distribution']])
        
        print(f"{key:<12} {'Теоретический':<15} {theo['n']:<6} {theo['delta']:<10.5f} {theo['cost']:<10.1f} [{theo_dist}]",file=fout)
        print(f"{key:<12} {'Монте-Карло':<15} {mc['n']:<6} {mc['delta']:<10.5f} {mc['cost']:<10.1f} [{mc_dist}]",file=fout)
    

    fout.close()
    return True

#==========================================
if __name__ == "__main__":
    process_fatigue_optimizer()
