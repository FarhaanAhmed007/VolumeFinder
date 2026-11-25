import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
import numpy as np
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Triple Integral Volume Solver", layout="wide")
st.title("Triple Integral Volume Solver & 3D Visualizer")
st.markdown("Enter your own custom limits in the sidebar or test with a preset shape.")

# Define symbolic variables
x, y, z, r, theta, rho, phi = sp.symbols('x y z r theta rho phi', real=True)
LOCAL_DICT = {
    'x': x, 'y': y, 'z': z,
    'r': r, 'theta': theta, 'rho': rho, 'phi': phi,
    'pi': sp.pi, 'sqrt': sp.sqrt, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'exp': sp.exp
}

# Allow standard math notation like x^2 and 2x
TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)

def parse_math(expr_str):
    return parse_expr(expr_str.strip(), local_dict=LOCAL_DICT, transformations=TRANSFORMATIONS)

# Presets
PRESETS = {
    "Custom Input (Type Your Own)": {
        "coord": "Cartesian (x, y, z)", "order": "z, y, x",
        "in_min": "0", "in_max": "4 - x^2 - y^2",
        "mid_min": "-sqrt(4 - x^2)", "mid_max": "sqrt(4 - x^2)",
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
    "Ice Cream Cone [Cylindrical]": {
        "coord": "Cylindrical (r, θ, z)", "order": "z, r, theta",
        "in_min": "r", "in_max": "sqrt(8 - r^2)",
        "mid_min": "0", "mid_max": "2",
        "out_min": "0", "out_max": "2*pi"
    },
    "Tetrahedron (First Octant) [Cartesian]": {
        "coord": "Cartesian (x, y, z)", "order": "z, y, x",
        "in_min": "0", "in_max": "1 - x - y",
        "mid_min": "0", "mid_max": "1 - x",
        "out_min": "0", "out_max": "1"
    }
}

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("1. Choose Mode / Preset")
    preset_choice = st.selectbox("Start from:", list(PRESETS.keys()))
    p = PRESETS[preset_choice]

    coord_options = ["Cartesian (x, y, z)", "Cylindrical (r, θ, z)", "Spherical (ρ, φ, θ)"]
    coord_sys = st.selectbox("Coordinate System", coord_options, index=coord_options.index(p["coord"]))

    if "Cartesian" in coord_sys:
        order_opts = ["z, y, x", "z, x, y", "y, z, x", "y, x, z", "x, z, y", "x, y, z"]
        jacobian = sp.Integer(1)
    elif "Cylindrical" in coord_sys:
        order_opts = ["z, r, theta", "r, z, theta", "z, theta, r"]
        jacobian = r
    else:
        order_opts = ["rho, phi, theta", "rho, theta, phi"]
        jacobian = rho**2 * sp.sin(phi)

    default_idx = order_opts.index(p["order"]) if p["order"] in order_opts else 0
    order_str = st.selectbox("Integration Order (Inner → Outer)", order_opts, index=default_idx)

    # Use a form so users can finish typing all 6 limits before solving
    with st.form("integral_input_form"):
        st.header("2. Enter Your Limits")
        st.caption("Supports natural math: `x^2`, `2x`, `sqrt(4-x^2)`, `pi`, `sin(phi)`")
        
        v_labels = [v.strip() for v in order_str.split(",")]
        
        c_in1, c_in2 = st.columns(2)
        in_min = c_in1.text_input(f"Inner min ({v_labels[0]})", value=p["in_min"])
        in_max = c_in2.text_input(f"Inner max ({v_labels[0]})", value=p["in_max"])

        c_mid1, c_mid2 = st.columns(2)
        mid_min = c_mid1.text_input(f"Middle min ({v_labels[1]})", value=p["mid_min"])
        mid_max = c_mid2.text_input(f"Middle max ({v_labels[1]})", value=p["mid_max"])

        c_out1, c_out2 = st.columns(2)
        out_min = c_out1.text_input(f"Outer min ({v_labels[2]})", value=p["out_min"])
        out_max = c_out2.text_input(f"Outer max ({v_labels[2]})", value=p["out_max"])

        density_str = st.text_input("Integrand f (leave as 1 for Volume)", value="1")
        submitted = st.form_submit_button("Calculate & Render 3D", type="primary", use_container_width=True)

# --- MAIN DISPLAY ---
col1, col2 = st.columns([1.1, 1])

try:
    v_names = [v.strip() for v in order_str.split(",")]
    v1, v2, v3 = [LOCAL_DICT[name] for name in v_names]

    a1 = parse_math(in_min)
    b1 = parse_math(in_max)
    a2 = parse_math(mid_min)
    b2 = parse_math(mid_max)
    a3 = parse_math(out_min)
    b3 = parse_math(out_max)
    f_expr = parse_math(density_str)

    integrand = sp.simplify(f_expr * jacobian)

    with col1:
        st.subheader("Integral Setup & Step-by-Step Solution")
        integral_latex = (
            rf"V = \int_{{{sp.latex(a3)}}}^{{{sp.latex(b3)}}} "
            rf"\int_{{{sp.latex(a2)}}}^{{{sp.latex(b2)}}} "
            rf"\int_{{{sp.latex(a1)}}}^{{{sp.latex(b1)}}} "
            rf"\left({sp.latex(integrand)}\right) \, d{sp.latex(v1)}\, d{sp.latex(v2)}\, d{sp.latex(v3)}"
        )
        st.latex(integral_latex)

        step1 = sp.integrate(integrand, (v1, a1, b1))
        step2 = sp.integrate(step1, (v2, a2, b2))
        final_vol = sp.simplify(sp.integrate(step2, (v3, a3, b3)))

        st.markdown("**Step 1 (Inner Integral):**")
        st.latex(rf"\int_{{{sp.latex(a1)}}}^{{{sp.latex(b1)}}} {sp.latex(integrand)} \, d{sp.latex(v1)} = {sp.latex(sp.simplify(step1))}")

        st.markdown("**Step 2 (Middle Integral):**")
        st.latex(rf"\int_{{{sp.latex(a2)}}}^{{{sp.latex(b2)}}} \left({sp.latex(sp.simplify(step1))}\right) d{sp.latex(v2)} = {sp.latex(sp.simplify(step2))}")

        st.markdown("**Step 3 (Final Volume):**")
        m1, m2 = st.columns(2)
        m1.metric("Exact Result", str(final_vol))
        try:
            num_val = float(final_vol.evalf())
            m2.metric("Decimal Value", f"{num_val:.6f}")
        except Exception:
            m2.metric("Decimal Value", "Symbolic")

    with col2:
        st.subheader("3D Bounding Surface Preview")
        fig = go.Figure()

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
            st.info("3D surface preview supports `z, y, x`, `z, r, theta`, and `rho, phi, theta`. Symbolic integration above works for all orders!")

        fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
            margin=dict(l=0, r=0, b=0, t=10),
            height=480
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Invalid math input. Check variable names match your chosen coordinate system. Details: {e}")
