import numpy as np
import statsmodels.api as sm
from scipy import optimize
from scipy import stats
import matplotlib.pyplot as plt

#Данные
y = np.array([1.7, -5.4, -4.0, -5.9, -1.6, 0.0, 0.6, 2.1, 0.1,
              -4.9, -3.5, 5.9, 8.5, 9.9, 13.3, 11.1, 14.4, 16.2])
x = np.arange(1, len(y) + 1)
n = len(y)

#двиг для экспоненты
shift = abs(min(y)) + 1
y_shifted = y + shift
ln_y = np.log(y_shifted)

#подгонка лучшей модели: полином 4-й степени (H0)
deg_best = 4
coeffs_best = np.polyfit(x, y, deg_best)
p_best = np.poly1d(coeffs_best)
y_pred_best = p_best(x)
resid_best = y - y_pred_best
SSE_best_data = np.sum(resid_best**2)
# оценка шума при H0
sigma_best = np.std(resid_best, ddof=deg_best+1)

# подгонка худшей модели: экспонента (H1)
# лог-линейная подгонка, как в твоём коде
X = sm.add_constant(x)
model_ln = sm.OLS(ln_y, X).fit()
c_hat = model_ln.params[0]
b_hat = model_ln.params[1]
a_hat = np.exp(c_hat)
# функция прогнозирования худшей модели (фиксируем shift)
def p_worst_func(x_arr, a, b):
    return a * np.exp(b * x_arr) - shift

p_worst = lambda xx: a_hat * np.exp(b_hat * xx) - shift
y_pred_worst = p_worst(x)
resid_worst = y - y_pred_worst
SSE_worst_data = np.sum(resid_worst**2)
# оценка шума при H1 (по остаткам экспоненты)
sigma_worst = np.std(resid_worst, ddof=2)  # ddof=2: оценка для a и b

print("Исходные SSE: best(pol4) = {:.4f}, worst(exp) = {:.4f}".format(SSE_best_data, SSE_worst_data))
print("Оценки sigma: sigma_best = {:.4f}, sigma_worst = {:.4f}".format(sigma_best, sigma_worst))

#вспомогательная функция: подгоняет обе модели к данным y_sim и возвращает T = SSE_w - SSE_b
def fit_both_and_stat(x_arr, y_sim):
    # 1) полином 4-й степени (переподгонка)
    coeffs_b = np.polyfit(x_arr, y_sim, deg_best)
    p_b = np.poly1d(coeffs_b)
    SSE_b = np.sum((y_sim - p_b(x_arr))**2)

    # 2) экспонента — подгоняем нелинейно: y = a*exp(b*x) - shift
    # начальные приближения: используем исходные a_hat, b_hat
    try:
        popt, _ = optimize.curve_fit(lambda xx, a, b: a * np.exp(b * xx) - shift,
                                     x_arr, y_sim, p0=[a_hat, b_hat], maxfev=10000)
        a_fit, b_fit = popt
        y_w = a_fit * np.exp(b_fit * x_arr) - shift
        SSE_w = np.sum((y_sim - y_w)**2)
    except Exception as e:
        # если подгонка экспоненты не удалась — возвращаем None, симуляция пропускается
        return None

    return SSE_w - SSE_b

#параметры симуляции
alpha = 0.05
n_sim = 5000
rng = np.random.default_rng(12345)

#симуляции при H0: генерируем из полинома (лучшая модель) с шумом sigma_best
T_H0 = []
skipped_H0 = 0
for i in range(n_sim):
    y_sim = p_best(x) + rng.normal(0, sigma_best, size=n)
    stat = fit_both_and_stat(x, y_sim)
    if stat is None:
        skipped_H0 += 1
        continue
    T_H0.append(stat)
T_H0 = np.array(T_H0)
print(f"Симуляций при H0: запрошено {n_sim}, реально учтено {len(T_H0)}, пропущено {skipped_H0}")

# критическое значение (1-alpha квантиль распределения T при H0)
T_crit = np.quantile(T_H0, 1 - alpha)
print(f"Критическое значение T_crit (уровень {alpha}): {T_crit:.4f}")

#симуляции при H1: генерируем из экспоненты (худшая модель) с шумом sigma_worst
T_H1 = []
skipped_H1 = 0
for i in range(n_sim):
    y_sim = p_worst(x) + rng.normal(0, sigma_worst, size=n)
    stat = fit_both_and_stat(x, y_sim)
    if stat is None:
        skipped_H1 += 1
        continue
    T_H1.append(stat)
T_H1 = np.array(T_H1)
print(f"Симуляций при H1: запрошено {n_sim}, реально учтено {len(T_H1)}, пропущено {skipped_H1}")

# мощность: доля T_H1 > T_crit
power = np.mean(T_H1 > T_crit)
print(f"\nОценка мощности теста (H0 = полином 4, H1 = экспонента): power ≈ {power:.4f}")

#опционально: посмотреть гистограммы T_H0 и T_H1
plt.figure(figsize=(10,4))
plt.hist(T_H0, bins=60, alpha=0.6, label='T при H0 (полином)', density=True)
plt.hist(T_H1, bins=60, alpha=0.6, label='T при H1 (экспонента)', density=True)
plt.axvline(T_crit, color='k', linestyle='--', label=f'T_crit={T_crit:.3f}')
plt.legend()
plt.xlabel('T = SSE_worst - SSE_best')
plt.ylabel('Плотность')
plt.title('Распределение статистики T при H0 и H1')
plt.show()
