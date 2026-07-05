import numpy as np
from scipy.optimize import curve_fit
from typing import List, Tuple, Dict
import json
from datetime import datetime
from dataclasses import dataclass


# ===========================================================
# МОДЕЛИ КРИВОЙ УСТАЛОСТИ
# ===========================================================

def sn_curve_model_ku0(sigma: np.ndarray, sigma_inf: float, C: float, m: float) -> np.ndarray:
    sigma_adj = sigma - sigma_inf
    sigma_adj = np.maximum(sigma_adj, 1e-6)
    z = 10 ** ((np.log10(C) - np.log10(sigma_adj)) / m)
    return z


def sn_curve_model_ku1(sigma: np.ndarray, sigma_inf: float, C: float, m: float) -> np.ndarray:
    sigma_adj = sigma - sigma_inf
    sigma_adj = np.maximum(sigma_adj, 1e-6)
    z = (np.log10(C) - np.log10(sigma_adj)) / m
    return z


def sigma_from_lgN(lgN: float, sigma_inf: float, C: float, m: float, ku: int) -> float:
    if ku == 0:
        z = sigma_inf + C * (lgN)**(-m)
    if ku == 1:
        z = sigma_inf + C * (10**lgN)**(-m)
    return z


# ===========================================================
# ВЫЧИСЛЕНИЕ ВЕСОВ
# ===========================================================

def compute_weights(ni: List[int], lgN: List[float], slgN: List[float], ku: int) -> np.ndarray:
    w = np.zeros(len(ni))
    for i in range(len(ni)):
        if ku == 0:
            w[i] = ni[i] * lgN[i] * lgN[i] / (slgN[i] * slgN[i])
        else:
            w[i] = ni[i] / (slgN[i] * slgN[i])
    return w


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


# ===========================================================
# ОСНОВНАЯ ФУНКЦИЯ ОЦЕНКИ
# ===========================================================

def estimate_fatigue_curve(cx: List[float], ni: List[int], 
                           lgN: List[float], slgN: List[float],
                           ku: int) -> Dict:
    
    cx = np.array(cx)
    lgN = np.array(lgN)
    
    sigma_inf_guess = 0.5 * cx[-1]
    logC_init, m_init = compute_initial_parameters(cx, lgN, sigma_inf_guess, ku)
    C_guess = 10**logC_init
    w = compute_weights(ni, lgN, slgN, ku)
    
    bounds_lower = (0.0, C_guess / 3.0, m_init / 3.0)
    bounds_upper = (cx[-1] - 0.5, C_guess * 3.0, m_init * 3.0)

    if ku==0:
        sn_curve_model=lambda sigma,sigma_inf,C,m: 10**((np.log10(C)-np.log10(sigma-sigma_inf))/m)
    if ku==1:
        sn_curve_model=lambda sigma,sigma_inf,C,m: (np.log10(C)-np.log10(sigma-sigma_inf))/m

    popt,pcov=curve_fit(sn_curve_model,cx,lgN,
                               p0=[sigma_inf_guess,C_guess,m_init],
                               bounds=(bounds_lower,bounds_upper),
                               maxfev=10000,ftol=1e-12,xtol=1e-12)
    
    sigma_inf_est, C_est, m_est = popt
    log10C_est = np.log10(C_est)
    y_pred = sn_curve_model(cx, sigma_inf_est, C_est, m_est)
    residuals = y_pred - lgN
    Q = np.sum(w * residuals**2) / np.sum(w)
    se_sigma_inf = np.sqrt(max(pcov[0, 0], 0))
    se_C = np.sqrt(max(pcov[1, 1], 0))
    se_m = np.sqrt(max(pcov[2, 2], 0))
    
    sigma_calc_list = []
    for i in range(len(cx)):
        sigma_calc_list.append(sigma_from_lgN(lgN[i], sigma_inf_est, C_est, m_est, ku))
    
    return {
        'success': True,
        'ku': ku,
        'sigma_inf': sigma_inf_est,
        'C': C_est,
        'm': m_est,
        'log10C': log10C_est,
        'Q': Q,
        'covariance': pcov,
        'std_errors': [se_sigma_inf, se_C, se_m],
        'residuals': residuals,
        'weights': w,
        'initial_params': {
            'sigma_inf': sigma_inf_guess,
            'C': C_guess,
            'm': m_init,
            'log10C': logC_init
        },
        'bounds': {'lower': bounds_lower, 'upper': bounds_upper},
        'y_pred': y_pred,
        'sigma_calc': sigma_calc_list
    }


def estimate_sigma_for_N0(N0_list: List[float], sigma_inf: float, C: float, m: float, ku: int) -> List[float]:
    results = []
    for N0 in N0_list:
        results.append(sigma_from_lgN(N0, sigma_inf, C, m, ku))
    return results

#=========================================================================

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

#=============================================================================================

def process_fatigue_trf() -> bool:

    inp_dir="Inp"
    out_dir="Out"
    inp_file="fatigue_trf.json"
    out_file="fatigue_trf.out" 
    fout=open(out_dir+"/"+out_file,'w')

    print("=" * 70,file=fout)
    print("ОЦЕНКА ПАРАМЕТРОВ КРИВОЙ УСТАЛОСТИ",file=fout)
    print("=" * 70,file=fout)

    with open(inp_dir+"/"+inp_file, 'r', encoding='windows-1251') as f:params_dict = json.load(f)

    params = MaterialParams.from_dict(params_dict)
    
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",file=fout)
    print(f"Уровней: {len(params.cx)}",file=fout)
    print(f"Образцов: {sum(params.ni)}",file=fout)

    results = {}

    if params.ku == 0 or params.ku == 2:
        print("\n" + "=" * 70,file=fout)
        print("РЕЗУЛЬТАТ ДЛЯ ku=0 (МОДЕЛЬ С lgN)",file=fout)
        print("=" * 70,file=fout)
        results[0] = estimate_fatigue_curve(params.cx, params.ni, params.lgN, params.slgN, 0)
        
        res = results[0]
        print("ВЕСА:",file=fout)
        for i, w in enumerate(res['weights']):
            print(f"  Weight[{i}] = {w:.2f}",file=fout)

        
        print("НАЧАЛЬНЫЕ ПАРАМЕТРЫ:",file=fout)
        print(f"  sigma_inf = {res['initial_params']['sigma_inf']:.4f} МПа (0.5 * S_min)",file=fout)
        print(f"  log10(C)  = {res['initial_params']['log10C']:.4f}",file=fout)
        print(f"  C         = {res['initial_params']['C']:.4f}",file=fout)
        print(f"  m         = {res['initial_params']['m']:.6f}",file=fout)
        
        print("ГРАНИЦЫ ПАРАМЕТРОВ:",file=fout)
        print(f"  sigma_inf: [{res['bounds']['lower'][0]:.2f}, {res['bounds']['upper'][0]:.2f}]",file=fout)
        print(f"  C:         [{res['bounds']['lower'][1]:.2f}, {res['bounds']['upper'][1]:.2f}]",file=fout)
        print(f"  m:         [{res['bounds']['lower'][2]:.2f}, {res['bounds']['upper'][2]:.2f}]",file=fout)
        
        print("СТАТУС: УСПЕШНО\n",file=fout)
        print("ФИНАЛЬНЫЕ ПАРАМЕТРЫ:",file=fout)
        print(f"  sigma_inf = {res['sigma_inf']:.2f} МПа",file=fout)
        print(f"  log10(C)  = {res['log10C']:.4f}",file=fout)
        print(f"  C         = {res['C']:.4f}",file=fout)
        print(f"  m         = {res['m']:.4f}",file=fout)
        print(f"  Q         = {res['Q']:.7f}",file=fout)
        
        print("СТАНДАРТНЫЕ ОШИБКИ:",file=fout)
        print(f"  se(sigma_inf) = {res['std_errors'][0]:.6e}",file=fout)
        print(f"  se(C)         = {res['std_errors'][1]:.6e}",file=fout)
        print(f"  se(m)         = {res['std_errors'][2]:.6e}",file=fout)
        
        print("КОВАРИАЦИОННАЯ МАТРИЦА:",file=fout)
        print("       [sigma_inf        C            m     ]",file=fout)
        for i in range(3):
            print(f"      [{res['covariance'][i,0]:12.6e} {res['covariance'][i,1]:12.6e} {res['covariance'][i,2]:12.6e}]",file=fout)
        
        print("СРАВНЕНИЕ:",file=fout)
        print(f"{'i':<3} {'S(МПа)':>10} {'lgN_exp':>12} {'lgN_calc':>12} {'S_calc(МПа)':>14} {'Невязка':>12}",file=fout)
        print("-" * 70,file=fout)
        for i in range(len(params.cx)):
            print(f"{i+1:<3} {params.cx[i]:10.1f} {params.lgN[i]:12.6f} {res['y_pred'][i]:12.6f} {res['sigma_calc'][i]:14.2f} {res['residuals'][i]:12.6e}",file=fout)

    if params.ku == 1 or params.ku == 2:
        print("\n" + "=" * 70,file=fout)
        print("РЕЗУЛЬТАТ ДЛЯ ku=1 (МОДЕЛЬ С N)",file=fout)
        print("=" * 70,file=fout)
        results[1] = estimate_fatigue_curve(params.cx, params.ni, params.lgN, params.slgN, 1)
        
        res = results[1]
        print("ВЕСА:",file=fout)
        for i, w in enumerate(res['weights']):
            print(f"  Weight[{i}] = {w:.2f}",file=fout)
        
        print("НАЧАЛЬНЫЕ ПАРАМЕТРЫ:",file=fout)
        print(f"  sigma_inf = {res['initial_params']['sigma_inf']:.4f} МПа (0.5 * S_min)",file=fout)
        print(f"  log10(C)  = {res['initial_params']['log10C']:.4f}",file=fout)
        print(f"  C         = {res['initial_params']['C']:.4f}",file=fout)
        print(f"  m         = {res['initial_params']['m']:.6f}",file=fout)
        
        print("ГРАНИЦЫ ПАРАМЕТРОВ:",file=fout)
        print(f"  sigma_inf: [{res['bounds']['lower'][0]:.2f}, {res['bounds']['upper'][0]:.2f}]",file=fout)
        print(f"  C:         [{res['bounds']['lower'][1]:.2f}, {res['bounds']['upper'][1]:.2f}]",file=fout)
        print(f"  m:         [{res['bounds']['lower'][2]:.2f}, {res['bounds']['upper'][2]:.2f}]",file=fout)
        
        print("СТАТУС: УСПЕШНО\n",file=fout)
        print("ФИНАЛЬНЫЕ ПАРАМЕТРЫ:",file=fout)
        print(f"  sigma_inf = {res['sigma_inf']:.2f} МПа",file=fout)
        print(f"  log10(C)  = {res['log10C']:.4f}",file=fout)
        print(f"  C         = {res['C']:.4f}",file=fout)
        print(f"  m         = {res['m']:.4f}",file=fout)
        print(f"  Q         = {res['Q']:.7f}",file=fout)
        
        print("СТАНДАРТНЫЕ ОШИБКИ:",file=fout)
        print(f"  se(sigma_inf) = {res['std_errors'][0]:.6e}",file=fout)
        print(f"  se(C)         = {res['std_errors'][1]:.6e}",file=fout)
        print(f"  se(m)         = {res['std_errors'][2]:.6e}",file=fout)
        
        print("КОВАРИАЦИОННАЯ МАТРИЦА:",file=fout)
        print("       [sigma_inf        C            m     ]",file=fout)
        for i in range(3):
            print(f"      [{res['covariance'][i,0]:12.6e} {res['covariance'][i,1]:12.6e} {res['covariance'][i,2]:12.6e}]",file=fout)
        
        print("СРАВНЕНИЕ:",file=fout)
        print(f"{'i':<3} {'S(МПа)':>10} {'lgN_exp':>12} {'lgN_calc':>12} {'S_calc(МПа)':>14} {'Невязка':>12}",file=fout)
        print("-" * 70,file=fout)
        for i in range(len(params.cx)):
            print(f"{i+1:<3} {params.cx[i]:10.1f} {params.lgN[i]:12.6f} {res['y_pred'][i]:12.6f} {res['sigma_calc'][i]:14.2f} {res['residuals'][i]:12.6e}",file=fout)

    if params.ku == 2:
        print("\n" + "=" * 70,file=fout)
        print("СРАВНЕНИЕ ДВУХ МОДЕЛЕЙ",file=fout)
        print("=" * 70,file=fout)
        print(f"{'Параметр':<15} {'ku=0':>20} {'ku=1':>20}",file=fout)
        print("-" * 55,file=fout)
        print(f"{'sigma_inf (МПа)':<15} {results[0]['sigma_inf']:20.2f} {results[1]['sigma_inf']:20.2f}",file=fout)
        print(f"{'C':<15} {results[0]['C']:20.2f} {results[1]['C']:20.2f}",file=fout)
        print(f"{'m':<15} {results[0]['m']:20.4f} {results[1]['m']:20.4f}",file=fout)
        print(f"{'Q':<15} {results[0]['Q']:20.7f} {results[1]['Q']:20.7f}",file=fout)

    print("\n" + "=" * 70,file=fout)
    print("АМПЛИТУДЫ НАПРЯЖЕНИЙ ДЛЯ БАЗОВЫХ ДОЛГОВЕЧНОСТЕЙ:",file=fout)
    print(f"{'i':<3} {'lg(N0)':>10} {'S(lgN0) (МПа)':>18}",file=fout)
    print("-" * 35,file=fout)

    if params.ku == 0 or params.ku == 2:
        sigma_list = estimate_sigma_for_N0(params.N0, results[0]['sigma_inf'], results[0]['C'], results[0]['m'], 0)
        for i, (N0, sigma) in enumerate(zip(params.N0, sigma_list)):
            print(f"{i+1:<3} {N0:10.3f} {sigma:18.7f}",file=fout)

    if params.ku == 1 or params.ku == 2:
        sigma_list = estimate_sigma_for_N0(params.N0, results[1]['sigma_inf'], results[1]['C'], results[1]['m'], 1)
        for i, (N0, sigma) in enumerate(zip(params.N0, sigma_list)):
            print(f"{i+1:<3} {N0:10.3f} {sigma:18.7f}",file=fout)

    print("\n" + "=" * 70,file=fout)
    fout.close()
    return True


if __name__ == "__main__":
    process_fatigue_trf()
