import numpy as np
from scipy.optimize import least_squares
from typing import List, Tuple, Dict
import json
from datetime import datetime
from dataclasses import dataclass


# ===========================================================
# МОДЕЛИ КРИВОЙ УСТАЛОСТИ (со штрафами)
# ===========================================================

def sn_curve_model(sigma: np.ndarray, sigma_inf: float, C: float, m: float, 
                   ku: int, cx_min: float) -> np.ndarray:
    if sigma_inf < 0 or sigma_inf > cx_min - 1.0 or m < 0.0 or m > 10.0 or C < 0:
        return np.full_like(sigma, 1e20)
    sigma_adj = sigma - sigma_inf
    sigma_adj = np.maximum(sigma_adj, 1e-6)
    if ku == 0:
        return 10**((np.log10(C) - np.log10(sigma_adj)) / m)
    else:
        return (np.log10(C) - np.log10(sigma_adj)) / m

#========================================================================================

def sigma_from_lgN(lgN: float, sigma_inf: float, C: float, m: float, ku: int) -> float:
    """вычисление напряжения по долговечности"""
    if ku == 0:
        return sigma_inf + C * (lgN) ** (-m)
    else:
        return sigma_inf + C * (10 ** lgN) ** (-m)

# ===========================================================
# НАЧАЛЬНЫЕ ПАРАМЕТРЫ
# ===========================================================

def compute_initial_parameters(cx: List[float], lgN: List[float], 
                                sigma_inf_guess: float, ku: int) -> Tuple[float, float]:
    s1 = np.log10(cx[0] - sigma_inf_guess)
    s2 = np.log10(cx[1] - sigma_inf_guess)
    if ku == 0:
        m_init = (s1 - s2) / (np.log10(lgN[1]) - np.log10(lgN[0]))
        logC_init = s1 + m_init * np.log10(lgN[0])
    else:
        m_init = (s1 - s2) / (lgN[1] - lgN[0])
        logC_init = s1 + m_init * lgN[0]
    return logC_init, m_init


# ====================== ТЕОРЕТИЧЕСКАЯ ОЦЕНКА ОШИБКИ ======================

def theoretical_std_sigma_R(sigma_inf: float, C: float, m: float, ku: int,
                            cx: List[float], lgN_levels: List[float], 
                            w: List[float], ni: List[int], N0: float) -> float:
    """
    Расчет теоретической стандартной ошибки предела выносливости sigma_R
    
    Параметры:
        sigma_inf, C, m - параметры кривой усталости
        ku - тип модели (0 или 1)
        cx - экспериментальные напряжения
        lgN_levels - экспериментальные логарифмы долговечностей
        w - весовая функция
        ni - количество образцов на каждом уровне
        N0 - базовая долговечность
    """
    n_i = np.array(ni)
    sigma_levels = np.array(cx)
    lgN_levels = np.array(lgN_levels)
    weights = np.array(w)
    
    sigma_adj = sigma_levels - sigma_inf
    sigma_adj = np.maximum(sigma_adj, 1e-6)
    ln10 = np.log(10)
    
    # Производные для матрицы чувствительности
    if ku==0:
        d_dsigma_inf = (1.0 / (m * C)) * ((sigma_adj / C) ** (-1.0/m - 1.0))
        d_dC = (1.0 / (m * C)) * ((sigma_adj / C) ** (-1.0/m))
        d_dm = ((sigma_adj / C) ** (-1.0/m)) * np.log(sigma_adj / C) * (1.0 / m**2)
    if ku==1:
        d_dsigma_inf = (1.0 / m) * (1.0 / (ln10 * sigma_adj))
        d_dC = (1.0 / m) * (1.0 / (ln10 * C))*np.ones_like(sigma_levels)
        d_dm = -(1.0 / m ** 2) * (np.log10(C) - np.log10(sigma_adj))
    
    J = np.column_stack([d_dsigma_inf, d_dC, d_dm])
    
    # Информационная матрица Фишера
    Fisher = J.T @ np.diag(weights) @ J
    
    if np.linalg.cond(Fisher) > 1e20: return 0.0
    
    try:
        Cov = np.linalg.inv(Fisher)
    except np.linalg.LinAlgError:
        return 0.0
    
    # Производные sigma_R по параметрам
    if ku == 0:
        # sigma_R = sigma_inf + C * N0^(-m)
        term = N0 ** (-m)
        dR_dsigma_inf = 1.0
        dR_dC = term
        dR_dm = -C * term * np.log(N0)
    else:
        # sigma_R = sigma_inf + C * (10^N0)^(-m) = sigma_inf + C * 10^(-m*N0)
        term = 10 ** (-m * N0)
        dR_dsigma_inf = 1.0
        dR_dC = term
        dR_dm = -C * term * np.log(10) * N0
    
    dR = np.array([dR_dsigma_inf, dR_dC, dR_dm])
    
    var_sigma_R = dR @ Cov @ dR
    return np.sqrt(max(var_sigma_R, 0.0))


# ===========================================================
# ФУНКЦИЯ ДЛЯ least_squares (взвешенный МНК)
# ===========================================================

def residuals(params: np.ndarray, sigma: np.ndarray, lgN: np.ndarray, 
              w: np.ndarray, ku: int, cx_min: float) -> np.ndarray:
    """Функция невязок для взвешенного МНК"""
    sigma_inf, C, m = params
    
    # Штрафы
    if sigma_inf < 0 or sigma_inf > cx_min - 1.0 or m < 0.0 or m > 10.0 or C < 0:
        return np.full_like(lgN, 1e20)
    
    sigma_adj = sigma - sigma_inf
    sigma_adj = np.maximum(sigma_adj, 1e-6)
    
    if ku == 0:
        y_pred =  10**((np.log10(C) - np.log10(sigma_adj)) / m)
    else:
        y_pred = (np.log10(C) - np.log10(sigma_adj)) / m
    
    # Взвешенные невязки
    return np.sqrt(w) * (y_pred - lgN)


# ===========================================================
# ОСНОВНАЯ ФУНКЦИЯ ОЦЕНКИ (ручной взвешенный МНК)
# ===========================================================

def estimate_fatigue_curve(cx: List[float], ni: List[int], 
                           lgN: List[float], w: List[float],
                           ku: int) -> Dict:
    
    cx = np.array(cx)
    lgN = np.array(lgN)
    w = np.array(w)

    sigma_inf_guess = 0.5 * cx[-1]
    logC_init, m_init = compute_initial_parameters(cx, lgN, sigma_inf_guess, ku)
    C_guess = 10 ** logC_init
    
    cx_min = cx[-1]
    
    # Взвешенный МНК через least_squares
    result = least_squares(
        residuals,
        x0=[sigma_inf_guess, C_guess, m_init],
        args=(cx, lgN, w, ku, cx_min),
        method='lm',
        max_nfev=10000,
        ftol=1e-12,
        xtol=1e-12
    )
    
    sigma_inf_est, C_est, m_est = result.x
    log10C_est = np.log10(C_est)
    
    # Предсказанные значения
    sigma_adj = cx - sigma_inf_est
    sigma_adj = np.maximum(sigma_adj, 1e-6)
    if ku == 0:
        y_pred = 10**((np.log10(C_est) - np.log10(sigma_adj)) / m_est)
    else:
        y_pred = (np.log10(C_est) - np.log10(sigma_adj)) / m_est
    
    residuals_arr = y_pred - lgN
    Q = np.sum(w * residuals_arr ** 2) / np.sum(w)
    
    # Аппроксимация ковариационной матрицы через якобиан
    try:
        eps = 1e-8
        J = []
        for i in range(3):
            x_plus = result.x.copy()
            x_minus = result.x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            r_plus = residuals(x_plus, cx, lgN, w, ku, cx_min)
            r_minus = residuals(x_minus, cx, lgN, w, ku, cx_min)
            deriv = (r_plus - r_minus) / (2 * eps)
            J.append(deriv)
        J = np.array(J).T
        JTJ = J.T @ J
        if np.linalg.cond(JTJ) < 1e30:
            pcov = np.linalg.inv(JTJ) * (np.sum(residuals_arr**2) / (len(cx) - 3))
        else:
            pcov = np.eye(3) * 1e10
        se_sigma_inf = np.sqrt(max(pcov[0, 0], 0))
        se_C = np.sqrt(max(pcov[1, 1], 0))
        se_m = np.sqrt(max(pcov[2, 2], 0))
    except:
        pcov = np.eye(3)
        se_sigma_inf = se_C = se_m = 0
    
    # Расчетные напряжения для сравнения
    sigma_calc_list = []
    for i in range(len(cx)):
        sigma_calc_list.append(sigma_from_lgN(lgN[i], sigma_inf_est, C_est, m_est, ku))
    
    return {
        'success': result.success,
        'ku': ku,
        'sigma_inf': sigma_inf_est,
        'C': C_est,
        'm': m_est,
        'log10C': log10C_est,
        'Q': Q,
        'covariance': pcov,
        'std_errors': [se_sigma_inf, se_C, se_m],
        'residuals': residuals_arr,
        'initial_params': {
            'sigma_inf': sigma_inf_guess,
            'C': C_guess,
            'm': m_init,
            'log10C': logC_init
        },
        'y_pred': y_pred,
        'sigma_calc': sigma_calc_list,
        'nfev': result.nfev,
        'ier': 1 if result.success else 0,
        'mesg': result.message
    }




# ================================================================

@dataclass
class MaterialParams:
    cx: List[float]
    lgN: List[float]
    slgN: List[float]
    ni: List[int]
    N0: List[float]
    ku: int

    @classmethod
    def from_dict(cls, data: dict) -> 'MaterialParams':
        return cls(
            cx=data['cx'],
            lgN=data['lgN'],
            slgN=data['slgN'],
            ni=data['ni'],
            N0=data['N0'],
            ku=data['ku']
        )


# ================================================================

def compute_weights(ni: List[int], lgN: List[float], slgN: List[float], ku: int) -> List[float]:
    w = []
    for i in range(len(ni)): w.append(ni[i] / (slgN[i]) ** 2)
        #if ku == 0:
        #    w.append(ni[i] * (lgN[i] / slgN[i]) ** 2)
        #else:
        #    w.append(ni[i] / (slgN[i]) ** 2)
    return w

#============================================================================================

def process_fatigue_mls_lm() -> bool:

    inp_dir="Inp"
    out_dir="Out"
    inp_file="fatigue_mls_lm.json"
    out_file="fatigue_mls_lm.out" 
    fout=open(out_dir+"/"+out_file,'w')

    print("=" * 80,file=fout)
    print("ОЦЕНКА ПАРАМЕТРОВ КРИВОЙ УСТАЛОСТИ (method='lm')",file=fout)
    print("=" * 80,file=fout)

    with open(inp_dir+"/"+inp_file, 'r', encoding='windows-1251') as f:params_dict = json.load(f)
    params = MaterialParams.from_dict(params_dict)
    
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",file=fout)
    print(f"Уровней: {len(params.cx)}",file=fout)
    print(f"Образцов: {sum(params.ni)}",file=fout)
    print("=" * 80,file=fout)

    results = {}

    # ========== РАСЧЕТ ДЛЯ ku=0 ==========
    if params.ku == 0 or params.ku == 2:
        print("\n" + "=" * 80,file=fout)
        print("РЕЗУЛЬТАТ ДЛЯ ku=0 (sigma = sigma_inf + C * (lgN)^(-m))",file=fout)
        print("=" * 80,file=fout)
        w0 = compute_weights(params.ni, params.lgN, params.slgN, 0)
        results[0] = estimate_fatigue_curve(params.cx, params.ni, params.lgN, w0, 0)
        res = results[0]
        print("ВЕСА:",file=fout)
        for i, weight in enumerate(w0):
            print(f"  Weight[{i}] = {weight:.4f}",file=fout)
        print("=" * 80,file=fout)

        
        print("НАЧАЛЬНЫЕ ПАРАМЕТРЫ:",file=fout)
        print(f"  sigma_inf = {res['initial_params']['sigma_inf']:.4f} МПа (0.5 * s_min)",file=fout)
        print(f"  log10(C)  = {res['initial_params']['log10C']:.4f}",file=fout)
        print(f"  C         = {res['initial_params']['C']:.4f}",file=fout)
        print(f"  m         = {res['initial_params']['m']:.6f}",file=fout)
        
        
        print("СТАТУС ОПТИМИЗАЦИИ:",file=fout)
        if res['success']:
            print(f"  УСПЕШНО (ier={res['ier']}: {res['mesg']})",file=fout)
        else:
            print(f"  НЕ УСПЕШНО (ier={res['ier']}: {res['mesg']})",file=fout)
        print(f"  Количество вызовов функции (nfev) = {res['nfev']}",file=fout)
        
        print("ФИНАЛЬНЫЕ ПАРАМЕТРЫ:",file=fout)
        print(f"  sigma_inf = {res['sigma_inf']:.7f} МПа",file=fout)
        print(f"  log10(C)  = {res['log10C']:.7f}",file=fout)
        print(f"  C         = {res['C']:.7f}",file=fout)
        print(f"  m         = {res['m']:.7f}",file=fout)
        print(f"  Q         = {res['Q']:.7f}",file=fout)
        
        print("СТАНДАРТНЫЕ ОШИБКИ ПАРАМЕТРОВ:",file=fout)
        print(f"  se(sigma_inf) = {res['std_errors'][0]:.6e}",file=fout)
        print(f"  se(C)         = {res['std_errors'][1]:.6e}",file=fout)
        print(f"  se(m)         = {res['std_errors'][2]:.6e}",file=fout)
        
        print("КОВАРИАЦИОННАЯ МАТРИЦА:",file=fout)
        print("       [sigma_inf        C            m     ]",file=fout)
        for i in range(3):
            print(f"      [{res['covariance'][i,0]:12.6e} {res['covariance'][i,1]:12.6e} {res['covariance'][i,2]:12.6e}]",file=fout)
        
        print("СРАВНЕНИЕ:",file=fout)
        print(f"{'i':<3} {'s(МПа)':>10} {'lgN_exp':>12} {'lgN_calc':>12} {'s_calc(МПа)':>14} {'Невязка':>12}",file=fout)
        print("-" * 70,file=fout)
        for i in range(len(params.cx)):
            print(f"{i+1:<3} {params.cx[i]:10.1f} {params.lgN[i]:12.6f} {res['y_pred'][i]:12.6f} {res['sigma_calc'][i]:14.2f} {res['residuals'][i]:12.6e}",file=fout)

    # ========== РАСЧЕТ ДЛЯ ku=1 ==========
    if params.ku == 1 or params.ku == 2:
        print("\n" + "=" * 80,file=fout)
        print("РЕЗУЛЬТАТ ДЛЯ ku=1 (sigma = sigma_inf + C * N^(-m))",file=fout)
        print("=" * 80,file=fout)
        w1 = compute_weights(params.ni, params.lgN, params.slgN, 1)
        results[1] = estimate_fatigue_curve(params.cx, params.ni, params.lgN, w1, 1)
        res = results[1]
        print("ВЕСА:",file=fout)
        for i, weight in enumerate(w1):
            print(f"  Weight[{i}] = {weight:.4f}",file=fout)
        
        print("НАЧАЛЬНЫЕ ПАРАМЕТРЫ:",file=fout)
        print(f"  sigma_inf = {res['initial_params']['sigma_inf']:.4f} МПа (0.5 * s_min)",file=fout)
        print(f"  log10(C)  = {res['initial_params']['log10C']:.4f}",file=fout)
        print(f"  C         = {res['initial_params']['C']:.4f}",file=fout)
        print(f"  m         = {res['initial_params']['m']:.6f}",file=fout)
        
        print("СТАТУС ОПТИМИЗАЦИИ:",file=fout)
        if res['success']:
            print(f"  УСПЕШНО (ier={res['ier']}: {res['mesg']})",file=fout)
        else:
            print(f"  НЕ УСПЕШНО (ier={res['ier']}: {res['mesg']})",file=fout)
        print(f"  Количество вызовов функции (nfev) = {res['nfev']}",file=fout)
        
        print("ФИНАЛЬНЫЕ ПАРАМЕТРЫ:",file=fout)
        print(f"  sigma_inf = {res['sigma_inf']:.7f} МПа",file=fout)
        print(f"  log10(C)  = {res['log10C']:.7f}",file=fout)
        print(f"  C         = {res['C']:.7f}",file=fout)
        print(f"  m         = {res['m']:.7f}",file=fout)
        print(f"  Q         = {res['Q']:.7f}",file=fout)
        
        print("СТАНДАРТНЫЕ ОШИБКИ ПАРАМЕТРОВ:",file=fout)
        print(f"  se(sigma_inf) = {res['std_errors'][0]:.6e}",file=fout)
        print(f"  se(C)         = {res['std_errors'][1]:.6e}",file=fout)
        print(f"  se(m)         = {res['std_errors'][2]:.6e}",file=fout)
        print()
        
        print("КОВАРИАЦИОННАЯ МАТРИЦА:",file=fout)
        print("       [sigma_inf        C            m     ]",file=fout)
        for i in range(3):
            print(f"      [{res['covariance'][i,0]:12.6e} {res['covariance'][i,1]:12.6e} {res['covariance'][i,2]:12.6e}]",file=fout)
        
        print("СРАВНЕНИЕ:",file=fout)
        print(f"{'i':<3} {'s(МПа)':>10} {'lgN_exp':>12} {'lgN_calc':>12} {'s_calc(МПа)':>14} {'Невязка':>12}",file=fout)
        print("-" * 70,file=fout)
        for i in range(len(params.cx)):
            print(f"{i+1:<3} {params.cx[i]:10.1f} {params.lgN[i]:12.6f} {res['y_pred'][i]:12.6f} {res['sigma_calc'][i]:14.2f} {res['residuals'][i]:12.6e}",file=fout)

    # ========== СРАВНЕНИЕ МОДЕЛЕЙ ==========
    if params.ku == 2:
        print("\n" + "=" * 80,file=fout)
        print("СРАВНЕНИЕ ДВУХ МОДЕЛЕЙ",file=fout)
        print("=" * 80,file=fout)
        print(f"{'Параметр':<15} {'ku=0':>20} {'ku=1':>20}",file=fout)
        print("-" * 55,file=fout)
        print(f"{'sigma_inf (МПа)':<15} {results[0]['sigma_inf']:20.2f} {results[1]['sigma_inf']:20.2f}",file=fout)
        print(f"{'C':<15} {results[0]['C']:20.2f} {results[1]['C']:20.2f}",file=fout)
        print(f"{'m':<15} {results[0]['m']:20.4f} {results[1]['m']:20.4f}",file=fout)
        print(f"{'Q':<15} {results[0]['Q']:20.7f} {results[1]['Q']:20.7f}",file=fout)
        print()

    # ========== НАПРЯЖЕНИЯ И ОШИБКИ ДЛЯ БАЗОВЫХ ДОЛГОВЕЧНОСТЕЙ ==========
    print("\n" + "=" * 80,file=fout)
    print("АМПЛИТУДЫ НАПРЯЖЕНИЙ И ОТНОСИТЕЛЬНЫЕ ОШИБКИ ДЛЯ БАЗОВЫХ ДОЛГОВЕЧНОСТЕЙ:",file=fout)
    print("=" * 80,file=fout)
    
    if params.ku == 0:
        print(f"{'i':<3} {'lg(N0)':>10} {'s(lgN0), МПа':>18} {'s_std, МПа':>14} {'delta = s_std/s':>14} {'delta,%':>10}",file=fout)
        print("-" * 80,file=fout)
        res = results[0]
        for i in range(len(params.N0)):
            sigma_R = res['sigma_inf'] + res['C'] * (params.N0[i]) ** (-res['m'])
            std_R = theoretical_std_sigma_R(
                res['sigma_inf'], res['C'], res['m'], 0,
                params.cx, params.lgN, w0, params.ni, params.N0[i]
            )
            delta = std_R / sigma_R if sigma_R > 0 else 0.0
            print(f"{i+1:<3} {params.N0[i]:10.3f} {sigma_R:18.7f} {std_R:14.7f} {delta:14.6f} {delta*100:9.2f}%",file=fout)
    
    elif params.ku == 1:
        print(f"{'i':<3} {'lg(N0)':>10} {'s(10^N0), МПа':>20} {'s_std, МПа':>14} {'delta = s_std/s':>14} {'delta,%':>10}",file=fout)
        print("-" * 80,file=fout)
        res = results[1]
        for i in range(len(params.N0)):
            sigma_R = res['sigma_inf'] + res['C'] * (10 ** params.N0[i]) ** (-res['m'])
            std_R = theoretical_std_sigma_R(
                res['sigma_inf'], res['C'], res['m'], 1,
                params.cx, params.lgN, w1,params.ni, params.N0[i]
            )
            delta = std_R / sigma_R if sigma_R > 0 else 0.0
            print(f"{i+1:<3} {params.N0[i]:10.3f} {sigma_R:20.7f} {std_R:14.7f} {delta:14.6f} {delta*100:9.2f}%",file=fout)
    
    elif params.ku == 2:
        # Вывод для обеих моделей
        print("\n--- МОДЕЛЬ ku=0 (sigma = sigma_inf + C * (lgN)^(-m)) ---",file=fout)
        print(f"{'i':<3} {'lg(N0)':>10} {'s(lgN0), МПа':>18} {'s_std, МПа':>14} {'delta = s_std/s':>14} {'delta,%':>10}",file=fout)
        print("-" * 80,file=fout)
        res0 = results[0]
        for i in range(len(params.N0)):
            sigma_R = res0['sigma_inf'] + res0['C'] * (params.N0[i]) ** (-res0['m'])
            std_R = theoretical_std_sigma_R(
                res0['sigma_inf'], res0['C'], res0['m'], 0,
                params.cx, params.lgN, w0, params.ni, params.N0[i]
            )
            delta = std_R / sigma_R if sigma_R > 0 else 0.0
            print(f"{i+1:<3} {params.N0[i]:10.3f} {sigma_R:18.7f} {std_R:14.7f} {delta:14.6f} {delta*100:9.2f}%",file=fout)
        
        print("\n--- МОДЕЛЬ ku=1 (sigma = sigma_inf + C * N^(-m)) ---",file=fout)
        print(f"{'i':<3} {'lg(N0)':>10} {'s(10^N0), МПа':>20} {'s_std, МПа':>14} {'delta = s_std/s':>14} {'delta,%':>10}",file=fout)
        print("-" * 80,file=fout)
        res1 = results[1]
        for i in range(len(params.N0)):
            sigma_R = res1['sigma_inf'] + res1['C'] * (10 ** params.N0[i]) ** (-res1['m'])
            std_R = theoretical_std_sigma_R(
                res1['sigma_inf'], res1['C'], res1['m'], 1,
                params.cx, params.lgN, w1, params.ni, params.N0[i]
            )
            delta = std_R / sigma_R if sigma_R > 0 else 0.0
            print(f"{i+1:<3} {params.N0[i]:10.3f} {sigma_R:20.7f} {std_R:14.7f} {delta:14.6f} {delta*100:9.2f}%",file=fout)

    fout.close()
    return True

if __name__ == "__main__":
    process_fatigue_mls_lm()
