#   Codigo que implementa el metodo de Runge-Kutta
#   de cuarto orden para resolver una ecuacion diferencial
#   
#           Autor:
#   Gilbert Alexander Mendez Cervera
#   mendezgilbert222304@outlook.com
#   Version 1.01 : 06/05/2025
#

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Definición de la EDO: dT/dx = -0.25(T - 25)
def f(x, T):
    return -0.25 * (T - 25)

# Método de Runge-Kutta de cuarto orden
def runge_kutta_4(f, x0, y0, x_end, h):
    x_vals = [x0]
    y_vals = [y0]

    x = x0
    y = y0

    while x < x_end:
        k1 = f(x, y)
        k2 = f(x + h/2, y + h/2 * k1)
        k3 = f(x + h/2, y + h/2 * k2)
        k4 = f(x + h, y + h * k3)

        y += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        x += h

        x_vals.append(round(x, 5))
        y_vals.append(y)

    return x_vals, y_vals

# Parámetros iniciales
x0 = 0
T0 = 100
x_end = 2
h = 0.1

# Solución con Runge-Kutta
x_vals, T_rk = runge_kutta_4(f, x0, T0, x_end, h)

# Solución exacta
T_exacta = [25 + 75 * np.exp(-0.25 * x) for x in x_vals]

# Guardar en CSV
df = pd.DataFrame({
    "x (m)": x_vals,
    "T_RK4 (°C)": T_rk,
    "T_Exacta (°C)": T_exacta
})
df.to_csv("transferencia_calor_rk4.csv", index=False)

# Graficar
plt.figure(figsize=(9, 5))
plt.plot(x_vals, T_rk, 'o-', label="RK4 (aproximada)", color="blue")
plt.plot(x_vals, T_exacta, '--', label="Solución exacta", color="red")
plt.xlabel("Distancia x (m)")
plt.ylabel("Temperatura T (°C)")
plt.title("Transferencia de Calor en un Tubo - RK4 vs Exacta")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("perfil_temperatura_rk4.png")
plt.show()

"""EJERCICIO #2
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del circuito
V = 10         # Voltaje en voltios
R = 1000       # Resistencia en ohmios
C = 0.001      # Capacitancia en faradios

# EDO: dq/dt = (V - q/C) / R
def f(t, q):
    return (V - q / C) / R

# Método de Runge-Kutta 4to orden
def runge_kutta_4(f, t0, q0, t_end, h):
    t_vals = [t0]
    q_vals = [q0]

    t = t0
    q = q0

    while t < t_end:
        k1 = f(t, q)
        k2 = f(t + h/2, q + h/2 * k1)
        k3 = f(t + h/2, q + h/2 * k2)
        k4 = f(t + h, q + h * k3)

        q += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        t += h

        t_vals.append(round(t, 5))
        q_vals.append(q)

    return t_vals, q_vals

# Condiciones iniciales
t0 = 0
q0 = 0
t_end = 1
h = 0.05

# Ejecutar Runge-Kutta
t_vals, q_vals = runge_kutta_4(f, t0, q0, t_end, h)

# Graficar
plt.figure(figsize=(8, 5))
plt.plot(t_vals, q_vals, 'o-', label='Carga q(t)', color='green')
plt.title('Carga de un Capacitor en un Circuito RC')
plt.xlabel('Tiempo (s)')
plt.ylabel('Carga (Coulombs)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("carga_capacitor_rc.png")
plt.show()

"""

"""EJERCICIO #3
import numpy as np
import matplotlib.pyplot as plt

# Definición del sistema de EDOs:
# dy1/dt = y2
# dy2/dt = -2*y2 - 5*y1
def sistema(t, y):
    y1, y2 = y
    dy1 = y2
    dy2 = -2 * y2 - 5 * y1
    return np.array([dy1, dy2])

# Método de Runge-Kutta 4º orden para sistemas
def runge_kutta_sistema(f, t0, y0, t_end, h):
    t_vals = [t0]
    y1_vals = [y0[0]]
    y2_vals = [y0[1]]
    
    t = t0
    y = np.array(y0, dtype=float)


    while t < t_end:
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        
        y += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        t += h
        
        t_vals.append(round(t, 5))
        y1_vals.append(y[0])
        y2_vals.append(y[1])

    return t_vals, y1_vals, y2_vals

# Condiciones iniciales
t0 = 0
y0 = [1, 0]     # y1(0) = 1 (posición), y2(0) = 0 (velocidad)
t_end = 5
h = 0.1

# Resolver el sistema
t_vals, y1_vals, y2_vals = runge_kutta_sistema(sistema, t0, y0, t_end, h)

# Gráfica de la trayectoria de la masa
plt.figure(figsize=(9, 5))
plt.plot(t_vals, y1_vals, 'b-', label='Posición y₁(t)')
plt.title('Dinámica de un Resorte Amortiguado')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición (m)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("trayectoria_resorte.png")
plt.show()
"""