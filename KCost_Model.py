#Type this into terminal: streamlit run doublebind_streamlit_KCost.py
import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy import exp

# Page setup
st.set_page_config(layout="wide")
st.title("Double Bind Evolution Model")

# --- Layout ---
left_col, right_col = st.columns([1, 2])  # Sliders on left, plot/output on right

# --- Left column: sliders ---
with left_col:
    st.header("Parameters")

    evolvability = st.slider("Evolvability", 0.0, 1.0, 0.5)
    gamma1 = st.slider("Gamma 1 (Drug 1 sensitivity ↑ from Drug 2 resistance)", 0.0, 0.5, 0.05)
    gamma2 = st.slider("Gamma 2 (Drug 2 sensitivity ↑ from Drug 1 resistance)", 0.0, 0.5, 0.05)
    beta1 = st.slider("Beta 1 (Cost of Drug 1 resistance)", 0.0, 0.5, 0.05)
    beta2 = st.slider("Beta 2 (Cost of Drug 2 resistance)", 0.0, 0.5, 0.05)
    b = st.slider("b (Drug potency scaling)", 0.0, 1.0, 0.3)
    time = st.slider("Simulation Time", 100, 2000, 1000)

# --- Fixed parameters ---
Kmax = 10000
r = 0.25
IC = [200, 0.01, 0.01]  # Initial conditions: pop, drug1 strat, drug2 strat

# --- Model function ---
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

    def K(u1, u2):
        return Kmax * exp(-(beta1 * u1 + beta2 * u2))

    mu = (1 + gamma2 * u2) * m1 / (1 + b * u1) + (1 + gamma1 * u1) * m2 / (1 + b * u2)
    dxdt = x * (r * (1 - x / K(u1, u2)) - mu)

    dG1dv = b * m1 * (gamma2 * u2 + 1) / (b * u1 + 1) ** 2 - gamma1 * m2 / (b * u2 + 1) - beta1 * r * x * exp(
        beta1 * u1 + beta2 * u2
    ) / Kmax
    dG2dv = b * m2 * (gamma1 * u1 + 1) / (b * u2 + 1) ** 2 - gamma2 * m1 / (b * u1 + 1) - beta2 * r * x * exp(
        beta1 * u1 + beta2 * u2
    ) / Kmax

    dv1dt = evolvability * dG1dv
    dv2dt = evolvability * dG2dv

    if u1 <= 0.001 and dv1dt < 0:
        dv1dt = 0
    if u2 <= 0.001 and dv2dt < 0:
        dv2dt = 0

    return [dxdt, dv1dt, dv2dt]

# --- Solve ODEs ---
t_span = (0, time)
t_eval = np.arange(time + 1)
sol = solve_ivp(evoLV, t_span, IC, method='BDF', t_eval=t_eval)
pop = sol.y.T

# --- Right column: plots + output ---
with right_col:
    st.header("Simulation Output")

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(pop[:, 0], label="Population", color="black", lw=2)
    ax[0].set_ylabel("Pop Size, x")
    ax[0].legend()
    ax[0].set_title("Population Dynamics")

    ax[1].plot(pop[:, 1], label="Drug 1 Resistance", color="red", lw=2)
    ax[1].plot(pop[:, 2], label="Drug 2 Resistance", color="blue", lw=2)
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Resistance Level")
    ax[1].legend()
    ax[1].set_title("Resistance Dynamics")

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Final Values")
    st.metric("Final Population Size", f"{pop[:, 0][-1]:.2f}")
    st.metric("Final Drug 1 Resistance", f"{pop[:, 1][-1]:.3f}")
    st.metric("Final Drug 2 Resistance", f"{pop[:, 2][-1]:.3f}")