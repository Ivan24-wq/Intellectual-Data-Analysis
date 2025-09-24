from math import comb

#Вероятность получить k промахов
def binom(n, k, p):
    return comb(n, k) * (p**k) * ((1 - p)**(n - k))

#Вероятность получить не болбьше k промахов
def ctf(n, k, p):
    return sum(binom(n, i, p) for i in range(0, k + 1))

#Поиск критических границ
#низняя граница
def find_critical_bonus(n, p0 = 0.2, alpha = 0.05):
    k_low = 0
    while ctf(n, k_low, p0) < alpha / 2:
        k_low += 1
    
    #Верхняя граница
    k_high = n
    while(1 - ctf(n, k_high - 1, p0)) < alpha / 2:
        k_high -= 1
    return k_low, k_high

#Запуск
n = 100
p0 = 0.2
for alpha in [0.1, 0.05, 0.01]:
    low, high = find_critical_bonus(n, p0, alpha)
    print(f"alpha = {alpha}: границы [{low}, {high}]")