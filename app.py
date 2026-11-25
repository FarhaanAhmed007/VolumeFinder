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

# Custom CSS for clean spacing, vertical steps, and oversized final volume card
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        border: 2.5px solid #38bdf8 !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 30px rgba(56, 189, 248, 0.25) !important;
        padding: 24px 32px !important;
        margin-top: 1rem !important;
    }
    .final-tag {
        color: #38bdf8;
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .final-decimal {
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 900;
        line-height: 1.15;
        margin: 0;
    }
    .step-badge {
        display: inline-block;
        background-color: #1e293b;
        color: #38bdf8;
        border: 1px solid #38bdf8;
        border-radius: 6px;
        padding: 4px 12px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Bar
h_left, h_right = st.columns([3, 1])
with h_left:
    st.markdown("## TRIPLE INTEGRAL VOLUME FINDER")
with h_right:
    st.markdown(
        "<div style='text-align:right; padding-top:14px; color:#94a3b8; font-weight:600; font-size:1.05rem;'>A project by Group 3</div>",
        unsafe_allow_html=True,
    )

# Symbolic variables
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
    st.header("1. Preset or Custom")
    preset_choice = st.selectbox("Select Example:", list(PRESETS.keys()))
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
        st.header("2. Limits of Integration")
        st.caption("Type naturally: `x^2`, `2x`, `sqrt(9-x^2)`, `pi`")

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

# --- MAIN CALCULATION & ORDERED FLOW ---
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

    # =========================================================
    # SECTION 1: 3D MODEL DEMONSTRATION FIRST
    # =========================================================
    st.subheader("1. Interactive 3D Bounded Solid")
    t1, t2, t3 = st.columns([1.8, 1, 0.8])
    with t1:
        view_angle = st.radio(
            "Camera Angle:",
            ["3D Isometric", "Top-Down (xy-Shadow)", "Side Elevation"],
            horizontal=True,
        )
    with t2:
        surface_opacity = st.slider("Opacity", min_value=0.3, max_value=1.0, value=0.88, step=0.05)
    with t3:
        show_wireframe = st.checkbox("Mesh Grid", value=True)

    fig = go.Figure()
    lighting_cfg = dict(ambient=0.55, diffuse=0.85, specular=0.45, roughness=0.35, fresnel=0.2)
    contour_cfg = dict(
        x=dict(show=show_wireframe, color="rgba(255,255,255,0.22)", width=1),
        y=dict(show=show_wireframe, color="rgba(255,255,255,0.22)", width=1),
        z=dict(show=show_wireframe, color="rgba(255,255,255,0.22)", width=1),
    )

    if "Cartesian" in coord_sys and v_names == ["z", "y", "x"]:
        u_vals = np.linspace(float(a3.evalf()), float(b3.evalf()), 65)
        v_param = np.linspace(0, 1, 65)
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
                lighting=lighting_cfg, contours=contour_cfg, name="Upper Surface",
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z (top): %{z:.2f}<extra></extra>"
            )
        )
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z_low, colorscale="Blues", opacity=max(0.25, surface_opacity - 0.25),
                showscale=False, lighting=lighting_cfg, contours=contour_cfg, name="Lower Surface",
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z (bottom): %{z:.2f}<extra></extra>"
            )
        )

    elif "Cylindrical" in coord_sys and v_names == ["z", "r", "theta"]:
        th_vals = np.linspace(float(a3.evalf()), float(b3.evalf()), 70)
        p_vals = np.linspace(0, 1, 55)
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
                lighting=lighting_cfg, contours=contour_cfg, name="Upper Surface",
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z (top): %{z:.2f}<extra></extra>"
            )
        )
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z_low, colorscale="Blues", opacity=max(0.25, surface_opacity - 0.25),
                showscale=False, lighting=lighting_cfg, contours=contour_cfg, name="Lower Surface",
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z (bottom): %{z:.2f}<extra></extra>"
            )
        )

    elif "Spherical" in coord_sys and v_names == ["rho", "phi", "theta"]:
        th_vals = np.linspace(float(a3.evalf()), float(b3.evalf()), 70)
        p_vals = np.linspace(0, 1, 55)
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
                x=X, y=Y, z=Z, colorscale="Viridis", opacity=surface_opacity,
                lighting=lighting_cfg, contours=contour_cfg, name="Spherical Shell",
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>"
            )
        )
    else:
        st.info("3D preview supports `z, y, x`, `z, r, theta`, and `rho, phi, theta`. Step-by-step calculation below works for all orders!")

    if view_angle == "Top-Down (xy-Shadow)":
        camera_eye = dict(x=0.01, y=0.01, z=2.1)
    elif view_angle == "Side Elevation":
        camera_eye = dict(x=2.1, y=0.01, z=0.15)
    else:
        camera_eye = dict(x=1.45, y=1.45, z=1.15)

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", backgroundcolor="rgba(15,23,42,0.5)", gridcolor="rgba(148,163,184,0.25)", zerolinecolor="#38bdf8"),
            yaxis=dict(title="Y", backgroundcolor="rgba(15,23,42,0.5)", gridcolor="rgba(148,163,184,0.25)", zerolinecolor="#38bdf8"),
            zaxis=dict(title="Z", backgroundcolor="rgba(15,23,42,0.5)", gridcolor="rgba(148,163,184,0.25)", zerolinecolor="#38bdf8"),
            aspectmode="data",
            camera=dict(eye=camera_eye),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=560,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    # =========================================================
    # SECTION 2: TRIPLE INTEGRAL SETUP
    # =========================================================
    st.divider()
    st.subheader("2. Triple Integral Setup")
    integral_latex = (
        rf"\Large V = \int_{{{sp.latex(a3)}}}^{{{sp.latex(b3)}}} "
        rf"\int_{{{sp.latex(a2)}}}^{{{sp.latex(b2)}}} "
        rf"\int_{{{sp.latex(a1)}}}^{{{sp.latex(b1)}}} "
        rf"\left({sp.latex(integrand)}\right) \, d{sp.latex(v1)}\, d{sp.latex(v2)}\, d{sp.latex(v3)}"
    )
    st.latex(integral_latex)

    # =========================================================
    # SECTION 3: STEP-BY-STEP SOLUTION (ONE AFTER ANOTHER)
    # =========================================================
    st.divider()
    st.subheader("3. Step-by-Step Evaluation")

    st.markdown(f'<span class="step-badge">Step 1 — Integrate with respect to {v_names[0]} (Inner Integral)</span>', unsafe_allow_html=True)
    st.latex(
        rf"\large I_1 = \int_{{{sp.latex(a1)}}}^{{{sp.latex(b1)}}} {sp.latex(integrand)} \, d{sp.latex(v1)} = {sp.latex(sp.simplify(step1))}"
    )

    st.markdown(f'<span class="step-badge">Step 2 — Integrate with respect to {v_names[1]} (Middle Integral)</span>', unsafe_allow_html=True)
    st.latex(
        rf"\large I_2 = \int_{{{sp.latex(a2)}}}^{{{sp.latex(b2)}}} \left({sp.latex(sp.simplify(step1))}\right) d{sp.latex(v2)} = {sp.latex(sp.simplify(step2))}"
    )

    st.markdown(f'<span class="step-badge">Step 3 — Integrate with respect to {v_names[2]} (Outer Integral)</span>', unsafe_allow_html=True)
    st.latex(
        rf"\large V = \int_{{{sp.latex(a3)}}}^{{{sp.latex(b3)}}} \left({sp.latex(sp.simplify(step2))}\right) d{sp.latex(v3)} = {sp.latex(final_vol)}"
    )

    # =========================================================
    # SECTION 4: BIG HIGHLIGHTED FINAL VOLUME AT THE BOTTOM
    # =========================================================
    st.divider()
    try:
        num_str = f"{float(final_vol.evalf()):.6f} cubic units"
    except Exception:
        num_str = "Symbolic Expression"

    with st.container(border=True):
        f1, f2 = st.columns(2)
        with f1:
            st.markdown('<div class="final-tag">Final Exact Volume</div>', unsafe_allow_html=True)
            st.latex(rf"\Huge V = {sp.latex(final_vol)}")
        with f2:
            st.markdown('<div class="final-tag">Decimal Approximation</div>', unsafe_allow_html=True)
            st.markdown(f'<p class="final-decimal">{num_str}</p>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"Invalid math input. Check variable names match your chosen coordinate system. Details: {e}")
