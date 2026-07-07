import numpy as np
from scipy.stats import norm
from scipy.optimize import least_squares
import json
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib
warnings.filterwarnings('ignore')

# ==================== БИБЛИОТЕКА СТАТИСТИЧЕСКИХ ФУНКЦИЙ ====================

def norm_cdf(z):
    return norm.cdf(z)

def norm_pdf(z):
    return np.exp(-0.5*z*z) / np.sqrt(2*np.pi)

def weibull_cdf(z):
    return 1.0 - np.exp(-np.exp(z))

def weibull_pdf(z):
    return np.exp(z - np.exp(z))

# ==================== КЛАСС ДЛЯ ДАННЫХ ====================

class DataContainer:
    def __init__(self):
        self.ts = ""
        self.klevel = 0
        self.x = []
        self.x_orig = []  # исходные значения (для графика)
        self.nsample = []
        self.nfailure = []
        self.p = []

# ==================== ФУНКЦИЯ НЕВЯЗОК ДЛЯ LM ====================

def UpDownResiduals(xsimpl, nesm):
    
    s1=0.0
    s2=0.0 
    for i in range(nesm.klevel):
        z = (nesm.x[i] - xsimpl[0]) / xsimpl[1]
        if nesm.ts == "Normal":
            p = norm_cdf(z)
            d = norm_pdf(z)
        if nesm.ts == "Weibull":
            p = weibull_cdf(z)
            d = weibull_pdf(z)
        if nesm.ts == "BS":
            p = norm_cdf(z)
            d = norm_pdf(z)
        if(p<=0 or p>=1): return(10000000.0)
        fiz=(nesm.p[i]-p)*nesm.nsample[i]*d/(p*(1.-p))
        s1=s1+fiz
        s2=s2+fiz*z
    return(s1,s2)

# ==================== КОВАРИАЦИОННАЯ МАТРИЦА ====================

def CovMatrixUpDown(ts, stepx, bint):
    pi = 3.1415926535898
    kv = 2
    if stepx <= 0 or np.isinf(stepx) or np.isnan(stepx):
        stepx = 0.5
    if np.isinf(bint) or np.isnan(bint):
        bint = 0.0
    if ts == "Normal" or ts == "BS":
        m = int(4.0 / stepx) + 1
    elif ts == "Weibull":
        m = int(3.0 / stepx) + 1
    else:
        m = int(4.0 / stepx) + 1
    
    m = min(m, 50)
    m1 = 2 * m
    s2 = 0.0
    kk = 0
    
    zcov = np.zeros(m1)
    pcov = np.zeros(m1)
    qcov = np.zeros(m1)
    dcov = np.zeros(m1)
    wcov = np.zeros(m1)
    
    for i in range(m):
        zcov[i] = bint - stepx * (m - i)
        zcov[i+m] = bint + stepx * i
    
    zcov = np.clip(zcov, -20, 20)
    
    for i in range(m1):
        if ts == "Normal" or ts == "BS":
            pcov[i] = norm_cdf(zcov[i])
            dcov[i] = np.exp(-0.5*zcov[i]*zcov[i]) / np.sqrt(2.0 * pi)
            qcov[i] = 1.0 - pcov[i]
        elif ts == "Weibull":
            pcov[i] = weibull_cdf(zcov[i])
            dcov[i] = weibull_pdf(zcov[i])
            qcov[i] = 1.0 - pcov[i]
        
        pcov[i] = np.clip(pcov[i], 1e-10, 1-1e-10)
        qcov[i] = np.clip(qcov[i], 1e-10, 1-1e-10)
        dcov[i] = np.clip(dcov[i], 1e-10, 10)
    
    j = 0
    for i in range(m1):
        s1 = 1.0
        if kk == 1:
            for i1 in range(j, i):
                if pcov[i1] > 0:
                    s1 *= qcov[i1] / pcov[i1]
        else:
            if pcov[i] < qcov[i]:
                for i1 in range(i, m1):
                    if pcov[i1] >= qcov[i1]:
                        break
                    if qcov[i1] > 0:
                        s1 *= pcov[i1] / qcov[i1]
            else:
                kk = 1
                j = i
        wcov[i] = s1
        s2 += wcov[i]
    
    if s2 == 0:
        s2 = 1
    
    v = np.zeros((kv, kv))
    f1 = np.zeros(kv)
    
    for i in range(m1):
        f1[0] = -dcov[i]
        f1[1] = -dcov[i] * zcov[i]
        
        if wcov[i] > 0 and pcov[i] > 0 and pcov[i] < 1:
            z2 = wcov[i] / (pcov[i] * pcov[i] * qcov[i] + 1e-10)
            for i1 in range(kv):
                for j1 in range(kv):
                    v[i1][j1] += f1[i1] * f1[j1] * z2
    
    v = v / (2.0 * s2)
    
    try:
        v_inv = np.linalg.inv(v)
        if np.any(np.diag(v_inv) < 0) or np.any(np.diag(v_inv) > 1e6):
            return np.eye(kv) * 100.0
        return v_inv
    except:
        return np.eye(kv) * 100.0

# ==================== ФУНКЦИЯ ДЛЯ ГРАФИКОВ ====================

def plot_quantiles(case_name, res, nesm):

    """Построение графика в координатах P - Xp"""
    
    #plot_dir = os.path.join(output_dir, "plots")
    #os.makedirs(plot_dir, exist_ok=True)
    
    # Эмпирические данные (в исходных координатах)
    p_emp = []
    x_emp = []
    
    for i in range(nesm.klevel):
        if nesm.nsample[i] > 0:
            p = nesm.nfailure[i] / nesm.nsample[i]
            if p <= 0:
                p_display = 1e-6
            elif p >= 1:
                p_display = 1 - 1e-6
            else:
                p_display = p
            p_emp.append(p_display)
            # Для графика используем исходные значения
            x_emp.append(nesm.x_orig[i])
    
    # Теоретические квантили
    quantiles = res['quantiles']
    p_theor = sorted(quantiles.keys())
    x_theor = [quantiles[p] for p in p_theor]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Теоретическая кривая
    ax.plot(x_theor, p_theor, 'b-', linewidth=2, label='Теоретическая кривая')
    
    # Эмпирические точки
    ax.scatter(x_emp, p_emp, color='red', s=80, zorder=5, 
              label='Эмпирические данные')
    
    # Подписи
    for i in range(len(x_emp)):
        if nesm.nsample[i] > 0:
            ax.annotate(f'n={nesm.nsample[i]}', 
                       (x_emp[i], p_emp[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    ax.set_xlabel('X (квантиль)', fontsize=12)
    ax.set_ylabel('P (вероятность)', fontsize=12)
    ax.set_title(f'{case_name}\nРаспределение: {nesm.ts}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Для Weibull и BS используем логарифмическую шкалу
    if nesm.ts == "Weibull" or nesm.ts == "BS":
        ax.set_xscale('log')
        ax.set_xlabel('X (квантиль, лог. шкала)', fontsize=12)
    
    plt.show()

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

def MLE_UpDown(config):

    nesm = DataContainer()
    dist_type = config.get('distribution')
    if dist_type == "Normal": nesm.ts = "Normal"
    if dist_type == "Weibull": nesm.ts = "Weibull"
    if dist_type == "BS": nesm.ts = "BS"
    
    nesm.klevel = len(config['levels'])
    nesm.x_orig = config['levels'].copy()  # Сохраняем исходные
    nesm.x = config['levels'].copy()
    nesm.nsample = config['nsample']
    nesm.nfailure = config['nfailure']
    
    # Для Weibull и BS преобразуем в логарифмы (только для расчета)
    if nesm.ts == "Weibull" or nesm.ts == "BS":
        for i in range(nesm.klevel):
            nesm.x[i] = np.log(nesm.x[i])
    
    # Оценки Диксона-Муда
    kfail = sum(nesm.nfailure)
    knon = 0
    nnon = []
    for i in range(nesm.klevel):
        j = nesm.nsample[i] - nesm.nfailure[i]
        nnon.append(j)
        knon += j
    
    ksigne = 1
    if kfail < knon:
        ksigne = -1
    
    s1 = s2 = s3 = 0.0
    for i in range(nesm.klevel):
        if ksigne == -1:
            s1 += i * nesm.nfailure[i]
            s2 += i * i * nesm.nfailure[i]
            s3 += nesm.nfailure[i]
        else:
            s1 += i * nnon[i]
            s2 += i * i * nnon[i]
            s3 += nnon[i]
    
    if s3 == 0:
        s3 = 1
    
    d = nesm.x[1] - nesm.x[0] if nesm.klevel > 1 else 1.0
    cp = nesm.x[0] + d * (s1/s3 + ksigne * 0.5)
    s4 = (s3 * s2 - s1 * s1) / (s3 * s3)
    s = 1.62 * d * (s4 + 0.029)
    
    # Вероятности разрушения
    nesm.p = []
    for i in range(nesm.klevel):
        if nesm.nsample[i] > 0:
            z = float(nesm.nfailure[i]) / float(nesm.nsample[i])
        else:
            z = 0.5
        nesm.p.append(z)
    
    x0 = [cp,s]
    
    # Минимизация методом LM
    result=least_squares(UpDownResiduals,x0,args=(nesm,),method='lm',max_nfev=1000,ftol=1e-15,xtol=1e-15)

    xsimpl = result.x.tolist()
    res=UpDownResiduals(xsimpl,nesm)
    q=res[0]**2+res[1]**2

    stepx = 0.5
    bint = 0.0
    for i in range(1, nesm.klevel):
        if nesm.x[i] > xsimpl[0]:
            stepx = (nesm.x[i] - nesm.x[i-1]) / xsimpl[1]
            bint = (xsimpl[0] - nesm.x[i-1]) / xsimpl[1]
            break
    
    v = CovMatrixUpDown(nesm.ts, stepx, bint)
    
    quantiles = {}
    # Для BS квантили считаем в исходных координатах (через exp)
    for p in config.get('quantiles', [0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.999]):
        if nesm.ts == "Normal":
            zp = norm.ppf(p)
            q_val = xsimpl[0] + zp * xsimpl[1]
        elif nesm.ts == "Weibull":
            zp = np.log(-np.log(1.0 - p))
            q_val = np.exp(xsimpl[0] + zp * xsimpl[1])
        elif nesm.ts == "BS":
            zp = norm.ppf(p)
            q_val = np.exp(xsimpl[0] + zp * xsimpl[1])
        else:
            zp = norm.ppf(p)
            q_val = xsimpl[0] + zp * xsimpl[1]
        quantiles[p] = q_val
    
    return {
        'distribution': nesm.ts,
        'initial': {'a': cp, 'sigma': s},
        'mle': {'a': xsimpl[0], 'sigma': xsimpl[1]},
        'function_value': q,
        'covariance': v.tolist(),
        'std_errors': [np.sqrt(v[0][0]), np.sqrt(v[1][1])],
        'quantiles': quantiles,
        'iterations': result.nfev if hasattr(result, 'nfev') else 0,
        'success': result.success,
        'nesm': nesm
    }

# ==================== ОСНОВНАЯ ПРОГРАММА ====================

def process_updown():

    inp_dir="Inp"
    out_dir="Out"
    inp_file="updown.json"
    out_file="updown.out" 

    inp_file=inp_dir+"/"+inp_file
    out_file=out_dir+"/"+out_file

    with open(inp_file, 'r', encoding='utf-8') as f: config = json.load(f)
    results = {}
    for case_name, case_config in config.items():
        print(f"\nОбработка: {case_name}")
        results[case_name] = MLE_UpDown(case_config)
    
    # Сохраняем результаты в .out
    with open(out_file, 'w', encoding='windows-1251') as out:
        out.write("="*70 + "\n")
        out.write("МЕТОД МАКСИМАЛЬНОГО ПРАВДОПОДОБИЯ (MLE)\n")
        out.write("Оценка параметров методом 'вверх-вниз'\n")
        out.write("="*70 + "\n\n")
        
        for case_name, res in results.items():
            out.write(f"{'='*70}\n")
            out.write(f"СЛУЧАЙ: {case_name}\n")
            out.write(f"Распределение: {res['distribution']}\n")
            out.write("-"*70 + "\n\n")
            
            out.write("НАЧАЛЬНЫЕ ПРИБЛИЖЕНИЯ (Диксон-Муд):\n")
            out.write(f"  a     = {res['initial']['a']:.8f}\n")
            out.write(f"  sigma = {res['initial']['sigma']:.8f}\n\n")
            
            out.write("ОЦЕНКИ МП:\n")
            out.write(f"  a     = {res['mle']['a']:.8f}\n")
            out.write(f"  sigma = {res['mle']['sigma']:.8f}\n")
            out.write(f"  Функция = {res['function_value']:.6e}\n\n")
            
            out.write("КОВАРИАЦИОННАЯ МАТРИЦА:\n")
            cov = res['covariance']
            out.write(f"  [{cov[0][0]:.8f}  {cov[0][1]:.8f}]\n")
            out.write(f"  [{cov[1][0]:.8f}  {cov[1][1]:.8f}]\n\n")
            
            out.write("СТАНДАРТНЫЕ ОШИБКИ:\n")
            out.write(f"  SE(a)     = {res['std_errors'][0]:.8f}\n")
            out.write(f"  SE(sigma) = {res['std_errors'][1]:.8f}\n\n")
            
            out.write("КВАНТИЛИ:\n")
            out.write("  p\t\tКвантиль\n")
            out.write("  " + "-"*50 + "\n")
            for p, q in res['quantiles'].items():
                out.write(f"  {p:.4f}\t\t{q:.8f}\n")
            
            out.write(f"\n  Итераций: {res['iterations']}\n")
            out.write(f"  Успешно: {res['success']}\n")
            out.write("-"*70 + "\n\n")
        
        out.write("="*70 + "\n")
        out.write("Расчет завершен\n")
        out.write("="*70 + "\n")
    
    # Построение графиков
    for case_name, res in results.items():
        if 'nesm' in res:
            plot_quantiles(case_name,res,res['nesm'])
    
    return results

# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    results = process_updown()