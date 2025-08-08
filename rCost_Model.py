#Type this into terminal: streamlit run doublebind_streamlit_rCost.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy import exp

st.title("Cancer Evolution: Double Bind Therapy Simulator")

# Sidebar for parameters
st.sidebar.header("Model Parameters")

# Fixed parameters
time = st.sidebar.slider("Simulation Time", 100, 2000, 1000)
pop = st.sidebar.number_input("Initial Population Size", 1, 10000, 200)
strat1 = st.sidebar.number_input("Initial Strategy 1 (Drug 1)", 0.0, 1.0, 0.01)
strat2 = st.sidebar.number_input("Initial Strategy 2 (Drug 2)", 0.0, 1.0, 0.01)

K = st.sidebar.number_input("Carrying Capacity (K)", 100, 100000, 10000)
r = st.sidebar.slider("Growth Rate (r)", 0.01, 1.0, 0.25)

evolvability = st.sidebar.slider("Evolvability", 0.0, 1.0, 0.5)
gamma = st.sidebar.slider("Double Bind Strength (γ1 = γ2)", 0.0, 1.0, 0.05)
beta = st.sidebar.slider("Cost of Resistance (β1 = β2)", 0.0, 1.0, 0.05)
b = st.sidebar.slider("Drug Efficacy Curve Sharpness (b)", 0.01, 2.0, 0.3)

# Initial conditions
IC = [pop, strat1, strat2]

def evoLV(t, X):
    if t > 600:
        m1 = 0
        m2 = 0.2
    elif t > 200:
        m1 = 0.2
        m2 = 0
    else:
        m1 = m2 = 0

    x, u1, u2 = X

    if x < 1:
        x = 0

    def grow(u1, u2):
        return exp(-(beta * u1 + beta * u2))

    mu = (1 + gamma * u2) * m1 / (1 + b * u1) + (1 + gamma * u1) * m2 / (1 + b * u2)
    dxdt = x * (r * grow(u1, u2) * (1 - x / K) - mu)

    dG1dv = b * m1 * (gamma * u2 + 1) / (b * u1 + 1)**2 - beta * r * (1 - x / K) * exp(-beta * u1 - beta * u2) - gamma * m2 / (b * u2 + 1)
    dG2dv = b * m2 * (gamma * u1 + 1) / (b * u2 + 1)**2 - beta * r * (1 - x / K) * exp(-beta * u1 - beta * u2) - gamma * m1 / (b * u1 + 1)

    dv1dt = evolvability * dG1dv
    dv2dt = evolvability * dG2dv

    if u1 <= .001 and dv1dt < 0:
        dv1dt = 0
    if u2 <= .001 and dv2dt < 0:
        dv2dt = 0

    return [dxdt, dv1dt, dv2dt]

# Solve system
t_span = (0, time)
t_eval = np.arange(time + 1)
sol = solve_ivp(evoLV, t_span, IC, method='BDF', t_eval=t_eval)
pop = sol.y.T

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax[0].plot(t_eval, pop[:, 0], label='Cancer Cell Pop.', color='black')
ax[0].set_ylabel("Population Size")
ax[0].legend()

ax[1].plot(t_eval, pop[:, 1], label='Strategy 1 (Drug 1)', color='red')
ax[1].plot(t_eval, pop[:, 2], label='Strategy 2 (Drug 2)', color='blue')
ax[1].set_ylabel("Resistance Strategy")
ax[1].set_xlabel("Time")
ax[1].legend()

st.pyplot(fig)

# Final values
st.markdown(f"**Final Population Size**: {pop[:, 0][-1]:.2f}")
st.markdown(f"**Final Resistance (Drug 1)**: {pop[:, 1][-1]:.3f}")
st.markdown(f"**Final Resistance (Drug 2)**: {pop[:, 2][-1]:.3f}")