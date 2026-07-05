import numpy as np
import json
from datetime import datetime
from scipy import stats
from scipy.stats import norm
from scipy import optimize
from scipy.optimize import minimize, minimize_scalar, least_squares
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt
from scipy import linalg

# Настройка шрифтов для matplotlib
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False


# ================================================================
# Прогресс бар
# ================================================================

def print_progress(current: int,total: int, prefix: str = '', suffix: str = '',
                   decimals: int = 1, length: int = 40, fill: str = '█'):

    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if current == total:
        print()


# ================================================================
# Классы для работы с данными
# ================================================================

@dataclass
class CensoredData:
    data_name: str
    data: List[float]
    censored: List[int]
    bootstrap: str
    gamma_find: str
    gamma: float
    quantiles: List[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> 'CensoredData':
        return cls(
            data_name=data.get('data_name', 'unknown'),
            data=data['data'],
            censored=data.get('censored', [0] * len(data['data'])),
            bootstrap=data.get('bootstrap','YES'),
            gamma_find=data.get('gamma_find','YES'),
            gamma=data.get('gamma',0),
            quantiles=data.get('quantiles', [0.01, 0.05, 0.5, 0.95, 0.99])
        )


@dataclass
class BootstrapResult:
    q_b: float
    ci_lower: float
    ci_upper: float

# ================================================================
# Аналитические функции для лог-нормального с порогом γ
# ================================================================

def NormalMinFunction (params, y, n, cp, cko):
    mu_log, sigma_log = params
    z = (y - mu_log) / sigma_log
    p = stats.norm.cdf(z)
    d = stats.norm.pdf(z)
    psi = d / (1 - p + 1e-10)
    s1 = np.sum(psi)
    s2 = np.sum(psi * z)
    c1 = cp - mu_log + sigma_log * s1 / n
    c2 = cko**2 + (cp - mu_log)**2 + sigma_log**2 * (s2 / n - 1)
    return np.array([c1, c2])

# ================================================================

def CovMatrixMleN(n, x, r, mu_log, sigma_log):
    s1 = 0
    s2 = 0
    s3 = 0
    k = np.sum(1 - np.array(r))
    
    for i in range(n):
        z = (x[i] - mu_log) / sigma_log
        p = stats.norm.cdf(z)
        d = stats.norm.pdf(z)
        psi = d / (1 - p + 1e-10)
        s1 += r[i] * psi * (psi - z)
        s2 += r[i] * psi * z * (z * (psi - z) - 1)
        s3 += r[i] * psi * (z * (psi - z) - 1)
    
    v = np.zeros((2, 2))
    v[0][0] = (k + s1) / n
    v[0][1] = s3 / n
    v[1][0] = v[0][1]
    v[1][1] = (2 * k + s2) / n
    
    try:
        v = linalg.inv(v)
    except:
        v = np.linalg.pinv(v)
    
    return v

# ================================================================
# Оценка параметров лог-нормального распределения с фиксированным γ
# ================================================================

def fit_fixed_loc(data: np.ndarray, censored: np.ndarray, fixed_loc: float) -> dict:

    log_data_all=np.log10(data-fixed_loc)
    log_data_all.sort()
    
    failure_mask = censored == 0
    failure_data = log_data_all[failure_mask]
    censored_data = log_data_all[censored == 1]
    
    # Начальные приближения по разрушенным образцам
    if len(failure_data) > 0:
        cp = np.mean(failure_data)      # среднее по разрушенным
        cko = np.std(failure_data)      # std по разрушенным
    else:
        cp = np.mean(log_data_all)
        cko = np.std(log_data_all)
    
    n=len(data)
    k=len(censored_data) 
    # Оптимизация методом Левенберга-Марквардта
    res = optimize.least_squares(
        NormalMinFunction,
        [cp, cko],
        method='lm',
        args=(censored_data,n-k,cp,cko),
        max_nfev=500,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12
    )
    
    mu_opt = res.x[0]
    sigma_opt = res.x[1]

    r_final = NormalMinFunction([mu_opt,sigma_opt],censored_data,n-k,cp,cko)
    Q_min = np.sum(r_final**2)
    
    # Ковариационная матрица
    cov_matrix=CovMatrixMleN(n,log_data_all,censored,mu_opt,sigma_opt)
    
    return {
        'mu_log': mu_opt,
        'sigma_log': sigma_opt,
        'loc': fixed_loc,
        'mu_log_init': cp,
        'sigma_log_init': cko,
        'Q_min': Q_min,
        'cov_matrix': cov_matrix,
        'success': res.success,
        'message': res.message,
        'nfev': res.nfev,
        'Status': res.status
    }

# ================================================================
# АВТОМАТИЧЕСКИЙ ПОИСК ОПТИМАЛЬНОГО ПОРОГА γ
# ================================================================

def find_optimal_gamma_lognormal(data: np.ndarray, censored: np.ndarray,
                               min_gamma_ratio: float = 0.0,max_gamma_ratio: float = 0.99,
                               tol: float = 1e-6,max_iter: int = 50) -> dict:
    """
    Автоматический поиск оптимального порога γ для лог-нормального распределения
    методом минимизации Q = (∂lnL/∂μ)² + (∂lnL/∂σ)²
    """
    
    failure_min = np.min(data[censored == 0])
    min_gamma = failure_min * min_gamma_ratio
    max_gamma = failure_min * max_gamma_ratio
    
    print(f"  Диапазон поиска γ: [{min_gamma:.2f}, {max_gamma:.2f}]")
    
    def objective(gamma):
        if gamma < min_gamma or gamma > max_gamma:  return 1e10
        res = fit_fixed_loc(data, censored, gamma)
        return res['Q_min']
    
    result = minimize_scalar(
        objective,
        bounds=(min_gamma, max_gamma),
        method='bounded',
        options={'xatol': tol, 'maxiter': max_iter}
    )
    
    optimal_gamma=result.x
    
    # Повторная оценка при оптимальном γ
    fit_result = fit_fixed_loc(data, censored, optimal_gamma)
    fit_result['optimal_gamma'] = optimal_gamma
    fit_result['gamma_search_converged'] = result.success
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

#========================================================================================

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
# ПАРАМЕТРИЧЕСКИЙ БУТСТРЕП
# ================================================================

def parametric_bootstrap(
    data: np.ndarray,
    censored: np.ndarray,
    mu_hat: float,
    s_hat: float,
    gamma: float,
    p: float,
    n_bootstrap: int,
    conf_level: float,
    random_seed: int
) -> BootstrapResult:

    np.random.seed(random_seed)
    n=len(data)
    alpha=1.-conf_level
    bootstrap=[]
    print(f"\n  Параметрический бутстреп для p={p}:")

#===================================================================    
    for i in range(n_bootstrap):
        if (i+1) % 50 == 0 or i==n_bootstrap - 1:
            print_progress(i+1,n_bootstrap,prefix=f' Итерация',length=30)
        u=np.random.uniform(0,1,n)
        t_boot=10**(mu_hat+s_hat*norm.ppf(u))+gamma
        result=fit_fixed_loc(t_boot,censored,gamma)
        mu_b=result['mu_log']
        s_b=result['sigma_log']
        q_b=mu_b+s_b*norm.ppf(p)
        bootstrap.append(q_b)
#===================================================================    
    bootstrap.sort()
    ci_lower=np.percentile(bootstrap,100*alpha/2)
    ci_upper=np.percentile(bootstrap,100*(1.-alpha/2))
    
    return BootstrapResult(q_b=q_b,ci_lower=ci_lower,ci_upper=ci_upper)

#=====================================================================

def nctlimit(n,beta,zp,t1,t2,t12):
    zb=stats.norm.ppf(beta)
    d=zp*np.sqrt(n)
    if t1==0:
        tlow=stats.nct.ppf(1-beta,n-1,d)
        tup=stats.nct.ppf(beta,n-1,d)
    else:
        f1x=t2/(n-1)
        f2x=2*t12/np.sqrt(n)
        f4x=1-f1x/2
        e3x=f4x**2-zb**2*f1x
        e11=f4x*d+zb**2*f2x/2
        e2x=d**2-zb**2*t1
        e44=np.sqrt(abs(e11**2-e2x*e3x))
        tlow=(e11-e44)/e3x
        tup=(e11+e44)/e3x
    return(tlow,tup)

#=================================================================

def show_distr(tdistr,nbegin,nend,x,y,xp,yp,xplow,yplow,xpup,ypup,grid_size, distr_name):
    
    if tdistr=="Weibull":
        p=[0.01,0.025,0.05,0.1,0.2,0.3,0.5,0.7,0.9,0.95,0.995]
        zp=np.log(np.log(1./(1.-np.asarray(p))))
    if tdistr=="Normal":
        p=[0.005,0.01,0.025,0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.9,0.95,0.975,0.99,0.995]
        zp=stats.norm.ppf(np.asarray(p),0,1)
    kp=len(p)
    ymin=zp[0]
    ymax=zp[kp-1]+0.5
    xmin=min(xplow)
    xmax=max(xpup)+0.5
   
    grid = np.linspace(xmin, xmax, grid_size) 
    #if nbegin==True:figure,axes=plt.subplots(figsize=(12, 10))
    if nbegin==True:figure,axes=plt.subplots()
    plt.plot(x,y,'r+',markersize=12,label=u'Выборка') 
    plt.plot(xp,yp,'black',lw=3,label=u'Distribution')
    plt.plot(xplow,yplow,'g-',lw=2,label=u'Xlow')
    plt.plot(xpup,ypup,'g-.',lw=2,label=u'Xup')
    
    for i in range(kp):
        xx=[xmin,xmax]
        yy=[zp[i],zp[i]] 
        plt.text (xmax,zp[i],str(100*p[i])+"%")
        plt.plot(xx,yy,'black',lw=1,label='',linestyle='dashed')
    plt.text(xmax,ymax,"P%")

    plt.grid(ls=':') 
    plt.xlabel(r'${X}$', fontsize=18) 
    plt.ylabel(r'${Zp}$', fontsize=18) 
    plt.xlim((xmin, xmax)) 
    plt.ylim((ymin, ymax)) 
    title = 'Distribution {}'.format(distr_name) 
    plt.title(title, fontsize=20) 
    
    if nbegin==True:axes.legend()
    if nbegin==True and nend==True:plt.show()


# ================================================================
# ОСНОВНАЯ ПРОГРАММА
# ================================================================

def process_estimation_lognorm():

    inp_dir="Inp"
    out_dir="Out"
    inp_file="estimation_lognorm.json"
    out_file="estimation_lognorm.out" 
    fout=open(out_dir+"/"+out_file,'w')
    
    print("=" * 80,file=fout)
    print("ОЦЕНКА ПАРАМЕТРОВ ЛОГ-НОРМАЛЬНОГО РАСПРЕДЕЛЕНИЯ",file=fout)
    print("С АВТОМАТИЧЕСКИМ ПОИСКОМ ПОРОГА gamma",file=fout)
    print("ДЛЯ ЦЕНЗУРИРОВАННЫХ УСТАЛОСТНЫХ ДАННЫХ",file=fout)
    print("=" * 80,file=fout)
    print(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",file=fout)
    
    # Загрузка данных
    with open(inp_dir+"/"+inp_file, 'r', encoding='windows-1251') as f:params_dict = json.load(f)
    
    data_params = CensoredData.from_dict(params_dict)
    data = np.array(data_params.data)
    censored = np.array(data_params.censored)
    bootstrap=data_params.bootstrap
    gamma_find=data_params.gamma_find
    gamma=data_params.gamma
    n=len(data)
    k=np.sum(censored == 0)
    beta=0.95
    
    if gamma_find=="NO" and gamma>=data[0]: 
        print("Превышен предел порога: ",gamma,file=fout)
        fout.close() 
        return FALSE

    print(f"Испытания: {data_params.data_name}",file=fout)
    print(f"Количество образцов: {n}",file=fout)
    print(f"  Разрушено: {k}",file=fout)
    print(f"  Цензурировано: {np.sum(censored == 1)}",file=fout)
    print(f"Диапазон данных: [{np.min(data):.2f}, {np.max(data):.2f}]",file=fout)
    print(f"log10 диапазон: [{np.log10(np.min(data)):.4f}, {np.log10(np.max(data)):.4f}]",file=fout)
    print(f"Минимальное разрушенное значение: {np.min(data[censored == 0]):.2f}",file=fout)
    
    if gamma_find=="NO": results=fit_fixed_loc(data,censored,gamma)

    if gamma_find=="YES":
    # Автоматический поиск оптимального порога γ
        print("=" * 80,file=fout)
        print("ПОИСК ОПТИМАЛЬНОГО ПОРОГА gamma",file=fout)
        print("=" * 80,file=fout)
        results=find_optimal_gamma_lognormal(data,censored,min_gamma_ratio=0.0,max_gamma_ratio=0.99,
        tol=1e-6,max_iter=50)
        gamma=results['optimal_gamma']
        print(f"  Поиск сошёлся: {results['gamma_search_converged']}",file=fout)

    print("\n" + "=" * 80,file=fout)
    print("ПАРАМЕТРЫ РАСПРЕДЕЛЕНИЯ",file=fout)
    print("=" * 80,file=fout)
    mu=results['mu_log']
    s=results['sigma_log']
    mu_init=results['mu_log_init']
    s_init=results['sigma_log_init']

    print(f"\n ПОРОГ gamma = {gamma:.2f} циклов",file=fout)
    print(f"\n  mu={mu:.6f} (среднее lg(N-gamma))",file=fout)
    print(f"  s={s:.6f} (стандартное отклонение lg(N-gamma))",file=fout)
    print(f"\n  Начальные приближения:",file=fout)
    print(f"    mu_init = {mu_init:.6f}",file=fout)
    print(f"    s_init = {s_init:.6f}",file=fout)
    print(f"\nОПТИМИЗАЦИЯ:",file=fout)
    print(f"  Q={results['Q_min']:.12e}",file=fout)
    print(f"  Число оценок функции: {results['nfev']}",file=fout)
    print(f"  Успех: {results['success']}",file=fout)
    print(f"  Сообщение: {results['message']}",file=fout)
    print("\n" + "=" * 80,file=fout)
    print("КОВАРИАЦИОННАЯ МАТРИЦА ПАРАМЕТРОВ:",file=fout)
    print("=" * 80,file=fout)
    cov = results['cov_matrix']
    print(f"\n  Cov(mu,mu) = {cov[0,0]:.6e}",file=fout)
    print(f"  Cov(mu,s) = {cov[0,1]:.6e}",file=fout)
    print(f"  Cov(s,mu) = {cov[1,0]:.6e}",file=fout)
    print(f"  Cov(s,s) = {cov[1,1]:.6e}",file=fout)
    t1=cov[0,0]
    t2=cov[1,1]
    t12=cov[0,1]
    
    
    # ================================================================
    # ОЦЕНКА КАПЛАНА-МЕЙЕРА С КВАНТИЛЯМИ
    # ================================================================
    
    print("\n" + "=" * 90,file=fout)
    print("ОЦЕНКА КАПЛАНА-МЕЙЕРА:",file=fout)
    print("=" * 90,file=fout)
    
    km_times, km_survival, km_cdf, km_var, km_se = kaplan_meier(data, censored)
    theo_cdf = norm.cdf((np.log10(km_times)-mu)/s)
    
    # Таблица с квантилями КМ
    print(f"{'№':<4} {'Наработка':>18} {'F(t)':>12} {'Ftheo(t)':>12}"
          f" {'КМ квантиль':>14} {'ДИ нижн.':>14} {'ДИ верхн.':>14}",file=fout)
    print("-" * 110,file=fout)
    
    q_km=[]
    cl=[]
    cu=[]
    for i in range(len(km_times)):
        p_val = km_cdf[i]
        q_km.append(np.log10(kaplan_meier_quantile(km_times,km_cdf,p_val)))
        ci = delta_method_km_quantile_ci(km_times, km_survival, km_var, p_val)
        cl.append(np.log10(ci['ci_lower']))
        cu.append(np.log10(ci['ci_upper']))
        print(f"{i+1:<4} {km_times[i]:18.2f} {km_cdf[i]:12.5f} {theo_cdf[i]:12.5f}"                           f"{q_km[i]:12.7f} {cl[i]:12.7f} {cu[i]:12.7f}",file=fout)

    show_distr("Normal",True,True,
    np.log10(km_times),norm.ppf(theo_cdf),
    cl,norm.ppf(theo_cdf),q_km,norm.ppf(theo_cdf),cu,norm.ppf(theo_cdf),
    grid_size=k,distr_name="Distr")

    p=data_params.quantiles
    zp=norm.ppf(p)
    kp=len(p)
    xp=[]
    xplow=[]
    xpup=[]
    xp_a=[]
    xplow_a=[]
    xpup_a=[]

#=================BootsTrap=================================================================

    if bootstrap=="YES":
        for i in range(kp):
            res=parametric_bootstrap(data,censored,mu,s,gamma,p[i],1000,beta,42)
            print("Xplow=",res.ci_lower)
            print("Xp=",res.q_b)
            print("Xplow=",res.ci_upper)
            xp.append(res.q_b)
            xplow.append(res.ci_lower)
            xpup.append(res.ci_upper)
        print("\n" + "=" * 80,file=fout)
        print("БУТСТРЕП ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ",file=fout)
        print("=" * 80,file=fout)
        print(f"\n{'p':^12} {'lg(Xplow)':^12} {'lg(Xp)':^12} {'lg(Xpup)':^12}",file=fout)
        print("-" * 80,file=fout)
        for i in range(kp):
            print(f"{p[i]:12.6f}{xplow[i]:12.6f}{xp[i]:12.6f}{xpup[i]:12.6f}",file=fout)
        show_distr("Normal",True,True,
               np.log10(km_times),norm.ppf(theo_cdf),
               xp,zp,xplow,zp,xpup,zp,
               grid_size=n,distr_name=r'$N({a=}$'+str(round(mu,4))+",${s=}$"+str(round(s,4))+")")

#=================================================================================================

    print("\n" + "=" * 80,file=fout)
    print("АНАЛИТИЧЕСКИЕ ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ",file=fout)
    print("=" * 80,file=fout)
    print(f"\n{'p':^12} {'lg(Xplow)':^12} {'lg(Xp)':^12} {'lg(Xpup)':^12}",file=fout)
    print("=" * 80,file=fout)
    for i in range(kp):
        tlow,tup=nctlimit(n,beta,zp[i],t1,t2,t12)
        xplow_a.append(mu+s*tlow/np.sqrt(n))
        xp_a.append(mu+s*zp[i]) 
        xpup_a.append(mu+s*tup/np.sqrt(n))
    for i in range(kp):
        print(f"{p[i]:12.6f}{xplow_a[i]:12.6f}{xp_a[i]:12.6f}{xpup_a[i]:12.6f}",file=fout)

    show_distr("Normal",True,True,
               np.log10(km_times),norm.ppf(km_cdf),
               xp_a,zp,xplow_a,zp,xpup_a,zp,
               grid_size=n,distr_name=r'$N({a=}$'+str(round(mu,4))+",${s=}$"+str(round(s,4))+")")

    fout.close()
    return True 

#=======================================================================================

if __name__ == "__main__":
    process_estimation_lognorm()
