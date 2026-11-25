import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Triple Integral Volume Solver", layout="wide")
st.title("Triple Integral Volume Solver & 3D Visualizer")
st.markdown("Compute exact symbolic volume and visualize 3D bounded regions across **Cartesian**, **Cylindrical**, and **Spherical** coordinates.")

# Define symbolic variables
x, y, z, r, theta, rho, phi = sp.symbols('x y z r theta rho phi', real=True)
LOCAL_DICT = {
    'x': x, 'y': y, 'z': z,
    'r': r, 'theta': theta, 'rho': rho, 'phi': phi,
    'pi': sp.pi, 'sqrt': sp.sqrt, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'exp': sp.exp
}

# Preset examples for quick demos
PRESETS = {
    "Custom Input": {
        "coord": "Cartesian (x, y, z)", "order": "z, y, x",
        "in_min": "0", "in_max": "4 - x**2 - y**2",
        "mid_min": "-sqrt(4 - x**2)", "mid_max": "sqrt(4 - x**2)",
        "out_min": "-2", "out_max": "2"
    },
    "Sphere (Radius = 3) [Spherical]": {
        "coord": "Spherical (ρ, φ, θ)", "order": "rho, phi, theta",
        "in_min": "0", "in_max": "3",
        "mid_min": "0", "mid_max": "pi",
        "out_min": "0", "out_max": "2*pi"
    },
    "Cylinder (R = 2, H = 5) [Cylindrical]": {
        "coord": "Cylindrical (r, θ, z)", "order": "z, r, theta",
        "in_min": "0", "in_max": "5",
        "mid_min": "0", "mid_max": "2",
        "out_min": "0", "out_max": "2*pi"
    },
    "Ice Cream Cone (z = r to z = sqrt(8 - r^2)) [Cylindrical]": {
        "coord": "Cylindrical (r, θ, z)", "order": "z, r, theta",
        "in_min": "r", "in_max": "sqrt(8 - r**2)",
        "mid_min": "0", "mid_max": "2",
        "out_min": "0", "out_max": "2*pi"
    },
    "Tetrahedron (x + y + z = 1, First Octant) [Cartesian]": {
        "coord": "Cartesian (x, y, z)", "order": "z, y, x",
        "in_min": "0", "in_max": "1 - x - y",
        "mid_min": "0", "mid_max": "1 - x",
        "out_min": "0", "out_max": "1"
    }
}

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("1. Load Preset or Customize")
    preset_choice = st.selectbox("Quick Demo Presets", list(PRESETS.keys()))
    p = PRESETS[preset_choice]

    coord_options = ["Cartesian (x, y, z)", "Cylindrical (r, θ, z)", "Spherical (ρ, φ, θ)"]
    coord_sys = st.selectbox("Coordinate System", coord_options, index=coord_options.index(p["coord"]))

    if "Cartesian" in coord_sys:
        order_opts = ["z, y, x", "z, x, y", "y, z, x", "y, x, z", "x, z, y", "x, y, z"]
        jacobian = sp.Integer(1)
        j_latex = "1"
    elif "Cylindrical" in coord_sys:
        order_opts = ["z, r, theta", "r, z, theta", "z, theta, r"]
        jacobian = r
        j_latex = "r"
    else:
        order_opts = ["rho, phi, theta", "rho, theta, phi"]
        jacobian = rho**2 * sp.sin(phi)
        j_latex = r"\rho^2 \sin\phi"

    default_idx = order_opts.index(p["order"]) if p["order"] in order_opts else 0
    order_str = st.selectbox("Integration Order (Inner → Outer)", order_opts, index=default_idx)

    st.header("2. Integration Limits")
    st.caption("Use `**` for powers (e.g. `x**2`), `sqrt()`, `pi`.")
    in_min = st.text_input("Inner Lower Limit", value=p["in_min"])
    in_max = st.text_input("Inner Upper Limit", value=p["in_max"])
    mid_min = st.text_input("Middle Lower Limit", value=p["mid_min"])
    mid_max = st.text_input("Middle Upper Limit", value=p["mid_max"])
    out_min = st.text_input("Outer Lower Limit", value=p["out_min"])
    out_max = st.text_input("Outer Upper Limit", value=p["out_max"])

    density_str = st.text_input("Integrand / Density f (1 for pure volume)", value="1")
    solve_btn = st.button("Calculate & Plot Volume", type="primary", use_container_width=True)

# --- MAIN DISPLAY ---
col1, col2 = st.columns([1.1, 1])

try:
    # Parse variables and limits
    v_names = [v.strip() for v in order_str.split(",")]
    v1, v2, v3 = [LOCAL_DICT[name] for name in v_names]

    a1 = sp.sympify(in_min, locals=LOCAL_DICT)
    b1 = sp.sympify(in_max, locals=LOCAL_DICT)
    a2 = sp.sympify(mid_min, locals=LOCAL_DICT)
    b2 = sp.sympify(mid_max, locals=LOCAL_DICT)
    a3 = sp.sympify(out_min, locals=LOCAL_DICT)
    b3 = sp.sympify(out_max, locals=LOCAL_DICT)
    f_expr = sp.sympify(density_str, locals=LOCAL_DICT)

    integrand = sp.simplify(f_expr * jacobian)

    with col1:
        st.subheader("Integral Formulation & Step-by-Step Solution")
        integral_latex = (
            rf"V = \int_{{{sp.latex(a3)}}}^{{{sp.latex(b3)}}} "
            rf"\int_{{{sp.latex(a2)}}}^{{{sp.latex(b2)}}} "
            rf"\int_{{{sp.latex(a1)}}}^{{{sp.latex(b1)}}} "
            rf"\left({sp.latex(integrand)}\right) \, d{sp.latex(v1)}\, d{sp.latex(v2)}\, d{sp.latex(v3)}"
        )
        st.latex(integral_latex)

        # Compute step-by-step
        step1 = sp.integrate(integrand, (v1, a1, b1))
        step2 = sp.integrate(step1, (v2, a2, b2))
        final_vol = sp.simplify(sp.integrate(step2, (v3, a3, b3)))

        st.markdown("**Step 1 (Inner Integral):**")
        st.latex(rf"\int_{{{sp.latex(a1)}}}^{{{sp.latex(b1)}}} {sp.latex(integrand)} \, d{sp.latex(v1)} = {sp.latex(sp.simplify(step1))}")

        st.markdown("**Step 2 (Middle Integral):**")
        st.latex(rf"\int_{{{sp.latex(a2)}}}^{{{sp.latex(b2)}}} \left({sp.latex(sp.simplify(step1))}\right) d{sp.latex(v2)} = {sp.latex(sp.simplify(step2))}")

        st.markdown("**Step 3 (Final Volume):**")
        m1, m2 = st.columns(2)
        m1.metric("Exact Symbolic Result", str(final_vol))
        try:
            num_val = float(final_vol.evalf())
            m2.metric("Decimal Approximation", f"{num_val:.6f}")
        except Exception:
            m2.metric("Decimal Approximation", "Symbolic")

    with col2:
        st.subheader("3D Bounding Surface Preview")
        fig = go.Figure()

        # Generate 3D mesh visualization based on coordinate system
        if "Cartesian" in coord_sys and v_names == ["z", "y", "x"]:
            u_vals = np.linspace(float(a3.evalf()), float(b3.evalf()), 45)
            v_param = np.linspace(0, 1, 45)
            U, V = np.meshgrid(u_vals, v_param)

            f_y_min = sp.lambdify(x, a2, "numpy")
            f_y_max = sp.lambdify(x, b2, "numpy")
            Y_min_grid = f_y_min(U)
            Y_max_grid = f_y_max(U)
            Y = Y_min_grid + V * (Y_max_grid - Y_min_grid)
            X = U

            f_z_min = sp.lambdify((x, y), a1, "numpy")
            f_z_max = sp.lambdify((x, y), b1, "numpy")
            Z_low = np.broadcast_to(f_z_min(X, Y), X.shape)
            Z_high = np.broadcast_to(f_z_max(X, Y), X.shape)

            fig.add_trace(go.Surface(x=X, y=Y, z=Z_high, colorscale="Viridis", opacity=0.85, name="Upper Bound"))
            fig.add_trace(go.Surface(x=X, y=Y, z=Z_low, colorscale="Cividis", opacity=0.65, showscale=False, name="Lower Bound"))

        elif "Cylindrical" in coord_sys and v_names == ["z", "r", "theta"]:
            th_vals = np.linspace(float(a3.evalf()), float(b3.evalf()), 50)
            p_vals = np.linspace(0, 1, 40)
            TH, P = np.meshgrid(th_vals, p_vals)

            f_r_min = sp.lambdify(theta, a2, "numpy")
            f_r_max = sp.lambdify(theta, b2, "numpy")
            R = f_r_min(TH) + P * (f_r_max(TH) - f_r_min(TH))

            X = R * np.cos(TH)
            Y = R * np.sin(TH)

            f_z_min = sp.lambdify((r, theta), a1, "numpy")
            f_z_max = sp.lambdify((r, theta), b1, "numpy")
            Z_low = np.broadcast_to(f_z_min(R, TH), X.shape)
            Z_high = np.broadcast_to(f_z_max(R, TH), X.shape)

            fig.add_trace(go.Surface(x=X, y=Y, z=Z_high, colorscale="Viridis", opacity=0.85, name="Upper Surface"))
            fig.add_trace(go.Surface(x=X, y=Y, z=Z_low, colorscale="Cividis", opacity=0.65, showscale=False, name="Lower Surface"))

        elif "Spherical" in coord_sys and v_names == ["rho", "phi", "theta"]:
            th_vals = np.linspace(float(a3.evalf()), float(b3.evalf()), 50)
            p_vals = np.linspace(0, 1, 40)
            TH, P = np.meshgrid(th_vals, p_vals)

            f_phi_min = sp.lambdify(theta, a2, "numpy")
            f_phi_max = sp.lambdify(theta, b2, "numpy")
            PHI = f_phi_min(TH) + P * (f_phi_max(TH) - f_phi_min(TH))

            f_rho_max = sp.lambdify((phi, theta), b1, "numpy")
            RHO = np.broadcast_to(f_rho_max(PHI, TH), TH.shape)

            X = RHO * np.sin(PHI) * np.cos(TH)
            Y = RHO * np.sin(PHI) * np.sin(TH)
            Z = RHO * np.cos(PHI)

            fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale="Plasma", opacity=0.85, name="Outer Shell"))
        else:
            st.info("3D surface preview supports standard orders (`z, y, x`, `z, r, theta`, `rho, phi, theta`). Symbolic integration above still works for all orders!")

        fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
            margin=dict(l=0, r=0, b=0, t=10),
            height=480
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Could not evaluate integral. Check your syntax (use `**` for exponents, e.g., `x**2`). Details: {e}")
