import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
import numpy as np
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="TRIPLE INTEGRAL VOLUME FINDER", layout="wide")

# Custom CSS for the Highlighted Final Volume Card
st.markdown(
    """
    <style>
    .volume-hero {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 2px solid #38bdf8;
        border-radius: 14px;
        padding: 20px 28px;
        margin: 12px 0 22px 0;
        box-shadow: 0 8px 24px rgba(56, 189, 248, 0.18);
    }
    .volume-label {
        color: #38bdf8;
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .volume-decimal {
        color: #f8fafc;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("TRIPLE INTEGRAL VOLUME FINDER")
st.subheader("A project by Group 3")

# Define symbolic variables
x, y, z, r, theta, rho, phi = sp.symbols("x y z r theta rho phi", real=True)
LOCAL_DICT = {
    "x": x, "y": y, "z": z,
    "r": r, "theta": theta, "rho": rho, "phi": phi,
    "pi": sp.pi, "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
}

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

def parse_math(expr_str):
    return parse_expr(expr_str.strip(), local_dict=LOCAL_DICT, transformations=TRANSFORMATIONS)

PRESETS = {
    "Custom Input (Type Your Own)": {
        "coord": "Cartesian (x, y, z)", "order": "z, y, x",
        "in_min": "0", "in_max": "4 - x^2 - y^2",
        "mid_min": "-sqrt(4 - x^2)", "mid_max": "sqrt(4 - x^2)",
        "out_min": "-2", "out_max": "2",
    },
    "Sphere (Radius = 3) [Spherical]": {
        "coord": "Spherical (ρ, φ, θ)", "order": "rho, phi, theta",
        "in_min": "0", "in_max": "3",
        "mid_min": "0", "mid_max": "pi",
        "out_min": "0", "out_max": "2*pi",
    },
    "Cylinder (R = 2, H = 5) [Cylindrical]": {
        "coord": "Cylindrical (r, θ, z)", "order": "z, r, theta",
        "in_min": "0", "in_max": "5",
        "mid_min": "0", "mid_max": "2",
        "out_min": "0", "out_max": "2*pi",
    },
    "Ice Cream Cone [Cylindrical]": {
        "coord": "Cylindrical (r, θ, z)", "order": "z, r, theta",
        "in_min": "r", "in_max": "sqrt(8 - r^2)",
        "mid_min": "0", "mid_max": "2",
        "out_min": "0", "out_max": "2*pi",
    },
    "Tetrahedron (First Octant) [Cartesian]": {
        "coord": "Cartesian (x, y, z)", "order": "z, y, x",
        "in_min": "0", "in_max": "1 - x - y",
        "mid_min": "0", "mid_max": "1 - x",
        "out_min": "0", "out_max": "1",
    },
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

    with st.form("integral_input_form"):
        st.header("2. Enter Your Limits")
        st.caption("Supports `x^2`, `2x`, `sqrt(4-x^2)`, `pi`, `sin(phi)`")

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

        density_str = st.text_input("Integrand f (1 for Volume)", value="1")
        submitted = st.form_submit_button("Calculate & Render 3D", type="primary", use_container_width=True)

# --- MAIN CALCULATION & DISPLAY ---
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

    step1 = sp.integrate(integrand, (v1, a1, b1))
    step2 = sp.integrate(step1, (v2, a2, b2))
    final_vol = sp.simplify(sp.integrate(step2, (v3, a3, b3)))

    try:
        num_str = f"{float(final_vol.evalf()):.6f} cubic units"
    except Exception:
        num_str = "Symbolic Expression"

    # 1. HIGHLIGHTED FINAL VOLUME BANNER AT THE TOP
    st.markdown('<div class="volume-hero">', unsafe_allow_html=True)
    hv1, hv2 = st.columns([1.2, 1])
    with hv1:
        st.markdown('<div class="volume-label">Exact Calculated Volume</div>', unsafe_allow_html=True)
        st.latex(rf"\Large V = {sp.latex(final_vol)}")
    with hv2:
        st.markdown('<div class="volume-label">Decimal Approximation</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="volume-decimal">{num_str}</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. FULL-WIDTH, INTERACTIVE 3D DEMONSTRATION
    st.subheader("Interactive 3D Solid Demonstration")
    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1, 1])
    with ctrl1:
        view_angle = st.radio(
            "Camera View:",
            ["3D Isometric", "Top-Down (xy-Projection)", "Side Profile"],
            horizontal=True,
        )
    with ctrl2:
        surface_opacity = st.slider("Surface Transparency", min_value=0.25, max_value=1.0, value=0.85, step=0.05)
    with ctrl3:
        show_wireframe = st.checkbox("Show Grid Contours", value=True)

    fig = go.Figure()
    contour_cfg = dict(
        x=dict(show=show_wireframe, color="rgba(255,255,255,0.25)", width=1),
        y=dict(show=show_wireframe, color="rgba(255,255,255,0.25)", width=1),
        z=dict(show=show_wireframe, color="rgba(255,255,255,0.25)", width=1),
    )

    if "Cartesian" in coord_sys and v_names == ["z", "y", "x"]:
        u_vals = np.linspace(float(a3.evalf()), float(b3.evalf()), 60)
        v_param = np.linspace(0, 1, 60)
        U, V = np.meshgrid(u_vals, v_param)

        f_y_min = sp.lambdify(x, a2, "numpy")
        f_y_max = sp.lambdify(x, b2, "numpy")
        Y_min_grid = np.broadcast_to(f_y_min(U), U.shape)
        Y_max_grid = np.broadcast_to(f_y_max(U), U.shape)
        Y = Y_min_grid + V * (Y_max_grid - Y_min_grid)
        X = U

        f_z_min = sp.lambdify((x, y), a1, "numpy")
        f_z_max = sp.lambdify((x, y), b1, "numpy")
        Z_low = np.broadcast_to(f_z_min(X, Y), X.shape)
        Z_high = np.broadcast_to(f_z_max(X, Y), X.shape)

        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z_high, colorscale="Turbo", opacity=surface_opacity,
                contours=contour_cfg, name="Ceiling Surface",
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z (upper): %{z:.2f}<extra></extra>"
            )
        )
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z_low, colorscale="Blues", opacity=max(0.25, surface_opacity - 0.2),
                showscale=False, contours=contour_cfg, name="Floor Surface",
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z (lower): %{z:.2f}<extra></extra>"
            )
        )

    elif "Cylindrical" in coord_sys and v_names == ["z", "r", "theta"]:
        th_vals = np.linspace(float(a3.evalf()), float(b3.evalf()), 65)
        p_vals = np.linspace(0, 1, 50)
        TH, P = np.meshgrid(th_vals, p_vals)

        f_r_min = sp.lambdify(theta, a2, "numpy")
        f_r_max = sp.lambdify(theta, b2, "numpy")
        R_min_g = np.broadcast_to(f_r_min(TH), TH.shape)
        R_max_g = np.broadcast_to(f_r_max(TH), TH.shape)
        R = R_min_g + P * (R_max_g - R_min_g)

        X = R * np.cos(TH)
        Y = R * np.sin(TH)

        f_z_min = sp.lambdify((r, theta), a1, "numpy")
        f_z_max = sp.lambdify((r, theta), b1, "numpy")
        Z_low = np.broadcast_to(f_z_min(R, TH), X.shape)
        Z_high = np.broadcast_to(f_z_max(R, TH), X.shape)

        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z_high, colorscale="Turbo", opacity=surface_opacity,
                contours=contour_cfg, name="Upper Surface",
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z (upper): %{z:.2f}<extra></extra>"
            )
        )
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z_low, colorscale="Blues", opacity=max(0.25, surface_opacity - 0.2),
                showscale=False, contours=contour_cfg, name="Lower Surface",
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z (lower): %{z:.2f}<extra></extra>"
            )
        )

    elif "Spherical" in coord_sys and v_names == ["rho", "phi", "theta"]:
        th_vals = np.linspace(float(a3.evalf()), float(b3.evalf()), 65)
        p_vals = np.linspace(0, 1, 50)
        TH, P = np.meshgrid(th_vals, p_vals)

        f_phi_min = sp.lambdify(theta, a2, "numpy")
        f_phi_max = sp.lambdify(theta, b2, "numpy")
        PHI_min_g = np.broadcast_to(f_phi_min(TH), TH.shape)
        PHI_max_g = np.broadcast_to(f_phi_max(TH), TH.shape)
        PHI = PHI_min_g + P * (PHI_max_g - PHI_min_g)

        f_rho_max = sp.lambdify((phi, theta), b1, "numpy")
        RHO = np.broadcast_to(f_rho_max(PHI, TH), TH.shape)

        X = RHO * np.sin(PHI) * np.cos(TH)
        Y = RHO * np.sin(PHI) * np.sin(TH)
        Z = RHO * np.cos(PHI)

        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z, colorscale="Plasma", opacity=surface_opacity,
                contours=contour_cfg, name="Outer Shell",
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>"
            )
        )
    else:
        st.info("3D surface preview supports `z, y, x`, `z, r, theta`, and `rho, phi, theta`. Step-by-step integration below works for all orders!")

    # Camera presets
    if view_angle == "Top-Down (xy-Projection)":
        camera_eye = dict(x=0.01, y=0.01, z=2.2)
    elif view_angle == "Side Profile":
        camera_eye = dict(x=2.2, y=0.01, z=0.2)
    else:
        camera_eye = dict(x=1.55, y=1.55, z=1.25)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X Axis", backgroundcolor="rgba(15,23,42,0.6)", gridcolor="rgba(148,163,184,0.25)"),
            yaxis=dict(title="Y Axis", backgroundcolor="rgba(15,23,42,0.6)", gridcolor="rgba(148,163,184,0.25)"),
            zaxis=dict(title="Z Axis", backgroundcolor="rgba(15,23,42,0.6)", gridcolor="rgba(148,163,184,0.25)"),
            aspectmode="data",
            camera=dict(eye=camera_eye),
        ),
        margin=dict(l=0, r=0, b=0, t=10),
        height=650,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    # 3. ORGANIZED STEP-BY-STEP DERIVATION BELOW
    st.divider()
    st.subheader("Integral Formulation & Step-by-Step Derivation")

    integral_latex = (
        rf"V = \int_{{{sp.latex(a3)}}}^{{{sp.latex(b3)}}} "
        rf"\int_{{{sp.latex(a2)}}}^{{{sp.latex(b2)}}} "
        rf"\int_{{{sp.latex(a1)}}}^{{{sp.latex(b1)}}} "
        rf"\left({sp.latex(integrand)}\right) \, d{sp.latex(v1)}\, d{sp.latex(v2)}\, d{sp.latex(v3)}"
    )
    st.latex(integral_latex)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.info("**Step 1: Inner Integral**")
        st.latex(rf"\int_{{{sp.latex(a1)}}}^{{{sp.latex(b1)}}} {sp.latex(integrand)} \, d{sp.latex(v1)}")
        st.latex(rf"= {sp.latex(sp.simplify(step1))}")
    with s2:
        st.info("**Step 2: Middle Integral**")
        st.latex(rf"\int_{{{sp.latex(a2)}}}^{{{sp.latex(b2)}}} \left({sp.latex(sp.simplify(step1))}\right) d{sp.latex(v2)}")
        st.latex(rf"= {sp.latex(sp.simplify(step2))}")
    with s3:
        st.success("**Step 3: Outer Integral (Final)**")
        st.latex(rf"\int_{{{sp.latex(a3)}}}^{{{sp.latex(b3)}}} \left({sp.latex(sp.simplify(step2))}\right) d{sp.latex(v3)}")
        st.latex(rf"= {sp.latex(final_vol)}")

except Exception as e:
    st.error(f"Invalid math input. Check variable names match your chosen coordinate system. Details: {e}")
