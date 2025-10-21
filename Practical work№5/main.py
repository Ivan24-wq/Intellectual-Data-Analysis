import numpy as np
import matplotlib.pyplot as plt
import math
import time

#Мой номер в журнале
k = 5
N = 500
dt = 2 * math.pi / 1000
L = k / 10
omega = 100 / k

#Генерация ряда
x = np.zeros(N)
x[0] = 0.0
x[1] = (-1)**k * dt
for i in range(N - 2):
    x[i + 2] = (x[i + 1] * (2 * dt * L * (1 - x[i]**2))
                - x[i] * (1 + dt ** 2 + dt * L * (1 - x[i]**2))
                + dt**2 * math.sin(omega * i * dt))
i_array = np.arange(N)

#Градиентный спуск
def model(params, i_idx):
    a0, a1, a2, b1, b2, om1, om2 = params
    z1 = om1 * i_idx * dt
    z2 = om2 * i_idx * dt
    return a0 + a1*np.cos(z1) + b1*np.sin(z1) + a2 * np.cos(z2) + b2 * np.sin(z2)

#Функция ошибки
def mse_loss(params, x_value, i_idx):
    pred = model(params, i_idx)
    err = pred - x_value
    return np.mean(err**2), err, pred

#Градиенты градиентного спуска
def grads_branch(params, x_value, i_idx):
    Nloc = x_value.size
    a0, a1, a2, b1, b2, om1, om2 = params
    z1 = om1 * i_idx * dt
    z2 = om2 * i_idx * dt
    cos1, sin1 = np.cos(z1), np.sin(z1)
    cos2, sin2 = np.cos(z2), np.sin(z2)
    pred = a0 + a1*cos1 + b1*sin1 + a2*cos2 + b2*sin2
    err = pred - x_value