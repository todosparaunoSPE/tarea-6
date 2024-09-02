# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 13:48:20 2024

@author: jperezr
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import base64
import scipy.stats as stats
from scipy.stats import t


# Función para mostrar PDF
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        pdf_data = f.read()
    b64 = base64.b64encode(pdf_data).decode('utf-8')  # Codificar a base64
    pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="700" height="500" frameborder="0"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Barra lateral con sección de ayuda
st.sidebar.header("Ayuda")
st.sidebar.write("""
1. **Ejercicio 7**: Se analiza la relación entre la cantidad de aditivo y el tiempo de secado, ajustando un polinomio de segundo grado.
2. **Ejercicio 8**: Se realiza una regresión múltiple para predecir el número de giros en función de dos elementos de aleación.
3. **Ejercicio 9**: Se ajustan curvas exponenciales y de Gompertz para analizar la tasa de dosis en función de la altitud.
4. **Ejercicio 10**: Se calcula el intervalo de confianza para una correlación.
5. **Visualización de PDF**: Se muestra un archivo PDF relacionado con los ejercicios.
""")

# Título principal
st.title("Estadística para la Investigación, Tarea-6")
st.write("Por: Javier Horacio Pérez Ricárdez")

# Ejercicio 7
st.subheader("Ejercicio 7")

# Datos
data_ej7 = pd.DataFrame({
    "Aditivo (g)": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "T secado (h)": [12, 10.5, 10, 8, 7, 8, 7.5, 8.5, 9]
})

st.subheader("Datos de Aditivo y Tiempo de Secado")
st.table(data_ej7)

# a) Diagrama de dispersión
st.subheader("a) Diagrama de Dispersión")
fig, ax = plt.subplots()
ax.scatter(data_ej7["Aditivo (g)"], data_ej7["T secado (h)"], color='blue', label='Datos')
ax.set_xlabel("Aditivo (g)")
ax.set_ylabel("T secado (h)")
ax.set_title("Diagrama de Dispersión")
st.pyplot(fig)

# b) Ajuste de un polinomio de segundo grado
st.subheader("b) Ajuste de un Polinomio de Segundo Grado")
X = data_ej7["Aditivo (g)"].values.reshape(-1, 1)
y = data_ej7["T secado (h)"].values
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)

# Mostrar ecuación
coefs = model.coef_
intercept = model.intercept_
st.write(f"Ecuación del polinomio ajustado: y = {intercept:.2f} + {coefs[1]:.2f}x + {coefs[2]:.2f}x²")

# Graficar el polinomio ajustado
x_range = np.linspace(0, 8, 100).reshape(-1, 1)
y_poly_pred = model.predict(poly.transform(x_range))

fig, ax = plt.subplots()
ax.scatter(data_ej7["Aditivo (g)"], data_ej7["T secado (h)"], color='blue', label='Datos')
ax.plot(x_range, y_poly_pred, color='red', label='Polinomio Ajustado')
ax.set_xlabel("Aditivo (g)")
ax.set_ylabel("T secado (h)")
ax.set_title("Ajuste de Polinomio de Segundo Grado")
ax.legend()
st.pyplot(fig)

# c) Predicción
st.subheader("c) Predicción")
additive = 6.5
prediction = model.predict(poly.transform([[additive]]))[0]
st.write(f"Predicción del tiempo de secado para 6.5 g de aditivo: {prediction:.2f} horas")

# Ejercicio 8
st.title("Ejercicio 8")

# Datos
data_ej8 = pd.DataFrame({
    "Número de giros (y)": [41, 49, 69, 65, 40, 50, 58, 57, 31, 36, 44, 57, 19, 31, 33, 43],
    "% de elemento A (x1)": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
    "% de elemento B (x2)": [5, 5, 5, 5, 10, 10, 10, 10, 15, 15, 15, 15, 20, 20, 20, 20]
})

st.subheader("Datos de Número de Giros y Elementos de Aleación")
st.table(data_ej8)

# Regresión de Mínimos Cuadrados
X_ej8 = data_ej8[["% de elemento A (x1)", "% de elemento B (x2)"]]
y_ej8 = data_ej8["Número de giros (y)"]
model_ej8 = LinearRegression().fit(X_ej8, y_ej8)

# Mostrar ecuación
coefs_ej8 = model_ej8.coef_
intercept_ej8 = model_ej8.intercept_
st.write(f"Ecuación del plano ajustado: y = {intercept_ej8:.2f} + {coefs_ej8[0]:.2f}x1 + {coefs_ej8[1]:.2f}x2")

# Predicción
st.subheader("Predicción")
x1, x2 = 2.5, 12
prediction_ej8 = model_ej8.predict([[x1, x2]])[0]
st.write(f"Predicción del número de giros para x1 = {x1} y x2 = {x2}: {prediction_ej8:.2f} giros")

# Ejercicio 9
st.title("Ejercicio 9")

# Datos
data_ej9 = pd.DataFrame({
    "Altitud (x)": [50, 450, 780, 1200, 4400, 4800, 5300],
    "Tasa de dosis (y)": [28, 30, 32, 36, 51, 58, 69]
})

st.subheader("Datos de Altitud y Tasa de Dosis")
st.table(data_ej9)

# a) Ajuste de una curva exponencial
st.subheader("a) Ajuste de una Curva Exponencial")
X_ej9 = data_ej9["Altitud (x)"].values
y_ej9 = data_ej9["Tasa de dosis (y)"].values
model_ej9 = LinearRegression().fit(X_ej9.reshape(-1, 1), np.log(y_ej9))

# Mostrar ecuación
a = model_ej9.intercept_
b = model_ej9.coef_[0]
st.write(f"Ecuación ajustada: ln(y) = {a:.9f} + {b:.9f}x")

# b) Estimación de Dosis a 3000 pies
st.subheader("b) Estimación de Dosis a 3000 Pies")
altitud_3000 = 3000
log_y_3000 = model_ej9.predict([[altitud_3000]])[0]
y_3000 = np.exp(log_y_3000)
st.write(f"Dosis estimada a 3000 pies: {y_3000:.9f}")

# c) Cambio de Ecuación y Reestimación
st.subheader("c) Cambio de Ecuación y Reestimación")
c_ej9 = -b
a_ej9 = np.exp(a)
st.write(f"Ecuación ajustada: y = {a_ej9:.9f} * e^(-{c_ej9:.9f}x)")
y_3000_recalc = a_ej9 * np.exp(-c_ej9 * altitud_3000)
st.write(f"Dosis reestimada a 3000 pies: {y_3000_recalc:.9f}")

# d) Ajuste de Curva de Gompertz
st.subheader("d) Ajuste de una Curva de Gompertz")
Y_gomp = np.log(np.log(y_ej9))
model_gomp = LinearRegression().fit(X_ej9.reshape(-1, 1), Y_gomp)

# Mostrar ecuación de Gompertz
a_gomp = model_gomp.coef_[0]
b_gomp = model_gomp.intercept_
st.write(f"Ecuación de Gompertz: ln(ln(y)) = {b_gomp:.9f} + {a_gomp:.9f}x")

# Graficar curvas ajustadas
x_range_ej9 = np.linspace(0, 6000, 100)
y_exp = np.exp(model_ej9.predict(x_range_ej9.reshape(-1, 1)))
y_gomp = np.exp(np.exp(model_gomp.predict(x_range_ej9.reshape(-1, 1))))

fig, ax = plt.subplots()
ax.scatter(data_ej9["Altitud (x)"], data_ej9["Tasa de dosis (y)"], color='blue', label='Datos')
ax.plot(x_range_ej9, y_exp, color='red', label='Curva Exponencial Ajustada')
ax.plot(x_range_ej9, y_gomp, color='green', label='Curva de Gompertz Ajustada')
ax.set_xlabel("Altitud (x)")
ax.set_ylabel("Tasa de Dosis (y)")
ax.set_title("Ajustes de Curvas Exponencial y de Gompertz")
ax.legend()
st.pyplot(fig)





# Ejercicio 10
st.title("Ejercicio 10")


# Título y enunciado del ejercicio
st.title("Intervalo de confianza del coeficiente de correlación")
st.write("""
Dado un coeficiente de correlación muestral r=0.7 para calificaciones de los cursos de probabilidad y estadística de 30 estudiantes, vamos a construir un intervalo de confianza del 95% para el coeficiente de correlación poblacional.
""")


# Parámetros del ejercicio
r = 0.7
n = 30
alpha = 0.05

# Transformación de Fisher
z = 0.5 * np.log((1 + r) / (1 - r))

# Error estándar
SE_z = 1 / np.sqrt(n - 3)

# Valor crítico para el intervalo de confianza del 95%
z_alpha_over_2 = stats.norm.ppf(1 - alpha / 2)

# Intervalo de confianza en la escala de z
CI_z_lower = z - z_alpha_over_2 * SE_z
CI_z_upper = z + z_alpha_over_2 * SE_z

# Transformación inversa de Fisher
CI_r_lower = (np.exp(2 * CI_z_lower) - 1) / (np.exp(2 * CI_z_lower) + 1)
CI_r_upper = (np.exp(2 * CI_z_upper) - 1) / (np.exp(2 * CI_z_upper) + 1)

# Mostrar resultados
st.title("Intervalo de Confianza para el Coeficiente de Correlación")
st.write(f"Coeficiente de correlación (r): {r}")
st.write(f"Número de estudiantes: {n}")
st.write(f"Nivel de confianza: {100 * (1 - alpha)}%")
st.write(f"Intervalo de confianza del 95% para el coeficiente de correlación poblacional:")
st.write(f"({CI_r_lower:.5f}, {CI_r_upper:.5f})")



## Visualizar PDF
#st.sidebar.subheader("Visualización de PDF")
#pdf_path = "tarea-6.pdf"  # Cambia esto por la ruta a tu archivo PDF
#show_pdf(pdf_path)


# Cargar el archivo PDF
pdf_file_path = "tarea-6.pdf"  # Cambia esto a la ruta de tu archivo PDF

# Enlace al visor de Google Docs
pdf_url = f"https://docs.google.com/gview?url=http://your-domain.com/{pdf_file_path}&embedded=true"

# Incrustar el PDF
st.markdown(f'<iframe src="{pdf_url}" width="700" height="500" frameborder="0"></iframe>', unsafe_allow_html=True)
