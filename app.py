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

# Custom CSS for clean spacing, vertical steps, and highlighted final volume card
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
        text-align: center;
        margin-bottom: 8px;
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
    "Piecewise": sp.Piecewise,
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
    "SpaceX Starship V3 Full Stack (124.4 m) [Cylindrical]": {
        "coord": "Cylindrical (r, θ, z)", "order": "r, theta, z",
        "in_min": "0",
        "in_max": "Piecewise((4.5, z <= 111.4), (4.5*sqrt((124.4 - z)/13), True))",
        "mid_min": "0", "mid_max": "2*pi",
        "out_min": "0", "out_max": "124.4",
    },
    "De Laval Rocket Nozzle (Hollow Bell) [Cylindrical]": {
        "coord": "Cylindrical (r, θ, z)", "order": "r, theta, z",
        "in_min": "0.8 + 0.2*(z - 4)^2", "in_max": "1.2 + 0.2*(z - 4)^2",
        "mid_min": "0", "mid_max": "2*pi",
        "out_min": "0", "out_max": "6",
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

def add_starship_v3_hardware(fig, opacity_val):
    """Adds detailed SpaceX Starship V3 aerodynamic & propulsion hardware to the 124.4m stack."""
    # 1. Windward Ceramic Heat Shield Tiles (Ship: z = 72.3m to 124.4m, azimuth -88° to +88°)
    z_ship = np.linspace(72.3, 124.4, 48)
    th_tiles = np.linspace(-np.deg2rad(88), np.deg2rad(88), 32)
    Z_t, TH_t = np.meshgrid(z_ship, th_tiles)
    R_t = np.where(Z_t <= 111.4, 4.54, 4.54 * np.sqrt(np.maximum(0.0, (124.4 - Z_t) / 13.0)))
    fig.add_trace(
        go.Surface(
            x=R_t * np.cos(TH_t), y=R_t * np.sin(TH_t), z=Z_t,
            colorscale=[[0, "#0f172a"], [1, "#1e293b"]],
            opacity=min(1.0, opacity_val + 0.1), showscale=False,
            name="TPS Heat Shield Tiles",
            hovertemplate="Heat Shield Tile<br>z: %{z:.1f} m<extra></extra>"
        )
    )

    # 2. Hot-Staging Truss Interstage Ring (z = 68.3m to 72.3m)
    th_ring = np.linspace(0, 2 * np.pi, 37)
    for z_lvl in [68.3, 72.3]:
        fig.add_trace(
            go.Scatter3d(
                x=4.58 * np.cos(th_ring), y=4.58 * np.sin(th_ring), z=np.full_like(th_ring, z_lvl),
                mode="lines", line=dict(color="#38bdf8", width=4),
                showlegend=False, hoverinfo="skip"
            )
        )
    for ang in np.linspace(0, 2 * np.pi, 18, endpoint=False):
        fig.add_trace(
            go.Scatter3d(
                x=[4.58 * np.cos(ang), 4.58 * np.cos(ang + 0.17)],
                y=[4.58 * np.sin(ang), 4.58 * np.sin(ang + 0.17)],
                z=[68.3, 72.3],
                mode="lines", line=dict(color="#94a3b8", width=3),
                showlegend=False, hoverinfo="skip"
            )
        )

    # 3. Four Ship Aero Flaps (2 Aft at ±90°, 2 Forward shifted leeward at ±110°)
    def add_flap(az_deg, z_bot, z_top, span_out, sweep_bot):
        az = np.deg2rad(az_deg)
        r_root_bot = 4.48 if z_bot <= 111.4 else 4.48 * np.sqrt(max(0.0, (124.4 - z_bot) / 13.0))
        r_root_top = 4.48 if z_top <= 111.4 else 4.48 * np.sqrt(max(0.0, (124.4 - z_top) / 13.0))
        r_tip_bot = 4.5 + span_out
        r_tip_top = 4.5 + span_out * 0.45
        z_tip_bot = z_bot + sweep_bot
        z_tip_top = z_top - 0.6

        pts_r = [r_root_bot, r_tip_bot, r_tip_top, r_root_top]
        pts_z = [z_bot, z_tip_bot, z_tip_top, z_top]
        # Build thin 3D prism for flap
        d_th = 0.035
        vx, vy, vz = [], [], []
        for sgn in [-1, 1]:
            for rr, zz in zip(pts_r, pts_z):
                vx.append(rr * np.cos(az + sgn * d_th))
                vy.append(rr * np.sin(az + sgn * d_th))
                vz.append(zz)
        fig.add_trace(
            go.Mesh3d(
                x=vx, y=vy, z=vz,
                i=[0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3],
                j=[1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4],
                k=[2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7],
                color="#334155", opacity=0.95, flatshading=True,
                name="Aero Flap", hovertemplate=f"Aero Flap ({az_deg}°)<extra></extra>"
            )
        )

    # Aft Flaps (180° apart, 17m tip-to-tip span -> ~4m beyond 4.5m radius)
    add_flap(90, 72.8, 84.2, span_out=4.0, sweep_bot=1.5)
    add_flap(-90, 72.8, 84.2, span_out=4.0, sweep_bot=1.5)
    # Forward Flaps (V3 140° leeward placement: ±110° from 0° windward center)
    add_flap(110, 114.5, 121.0, span_out=2.6, sweep_bot=1.2)
    add_flap(-110, 114.5, 121.0, span_out=2.6, sweep_bot=1.2)

    # 4. Three V3 Booster Grid Fins (90° / 90° / 180° T-layout near z = 64m)
    for gf_deg in [0, 90, -90]:
        az = np.deg2rad(gf_deg)
        r0, r1 = 4.5, 7.6
        z0, z1 = 63.0, 65.2
        w_ang = 0.14
        vx = [r0*np.cos(az-w_ang), r1*np.cos(az-w_ang), r1*np.cos(az+w_ang), r0*np.cos(az+w_ang),
              r0*np.cos(az-w_ang), r1*np.cos(az-w_ang), r1*np.cos(az+w_ang), r0*np.cos(az+w_ang)]
        vy = [r0*np.sin(az-w_ang), r1*np.sin(az-w_ang), r1*np.sin(az+w_ang), r0*np.sin(az+w_ang),
              r0*np.sin(az-w_ang), r1*np.sin(az-w_ang), r1*np.sin(az+w_ang), r0*np.sin(az+w_ang)]
        vz = [z0, z0, z0, z0, z1, z1, z1, z1]
        fig.add_trace(
            go.Mesh3d(
                x=vx, y=vy, z=vz,
                i=[0, 0, 4, 4, 0, 0, 2, 2],
                j=[1, 2, 5, 6, 1, 5, 3, 7],
                k=[2, 3, 6, 7, 5, 4, 7, 6],
                color="#64748b", opacity=0.92,
                name="Grid Fin", hovertemplate=f"V3 Grid Fin ({gf_deg}°)<extra></extra>"
            )
        )

    # 5. Super Heavy 33 Raptor Engine Cluster at Base (z = 0 to 2.8m)
    engine_centers = [(0.0, 0.0)]
    for k in range(3):
        a = 2 * np.pi * k / 3
        engine_centers.append((1.15 * np.cos(a), 1.15 * np.sin(a)))
    for k in range(10):
        a = 2 * np.pi * k / 10
        engine_centers.append((2.45 * np.cos(a), 2.45 * np.sin(a)))
    for k in range(20):
        a = 2 * np.pi * k / 20
        engine_centers.append((3.75 * np.cos(a), 3.75 * np.sin(a)))

    # Draw outer ring of 20 Raptor bells + center cluster cleanly
    th_e = np.linspace(0, 2 * np.pi, 14)
    z_e = np.linspace(0.0, 2.6, 4)
    TH_e, Z_e = np.meshgrid(th_e, z_e)
    R_bell = 0.62 - 0.14 * Z_e
    for cx, cy in engine_centers[::2]:  # Render representative high-detail bell array
        fig.add_trace(
            go.Surface(
                x=cx + R_bell * np.cos(TH_e),
                y=cy + R_bell * np.sin(TH_e),
                z=Z_e,
                colorscale=[[0, "#f97316"], [1, "#334155"]],
                opacity=0.95, showscale=False,
                hoverinfo="skip"
            )
        )

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
        order_opts = ["z, r, theta", "r, theta, z", "r, z, theta", "z, theta, r"]
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
    # SECTION 1: CLEAN 3D MODEL DEMONSTRATION FIRST
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
        surface_opacity = st.slider("Opacity", min_value=0.25, max_value=1.0, value=0.88, step=0.05)
    with t3:
        show_wireframe = st.checkbox("Clean Mesh Lines", value=True)

    fig = go.Figure()
    lighting_cfg = dict(ambient=0.62, diffuse=0.85, specular=0.5, roughness=0.3, fresnel=0.25)
    contour_cfg = dict(
        x=dict(show=show_wireframe, color="rgba(255, 255, 255, 0.18)", width=1),
        y=dict(show=show_wireframe, color="rgba(255, 255, 255, 0.18)", width=1),
        z=dict(show=show_wireframe, color="rgba(255, 255, 255, 0.18)", width=1),
    )

    is_starship = "SpaceX Starship" in preset_choice

    if "Cartesian" in coord_sys and v_names == ["z", "y", "x"]:
        u_vals = np.linspace(float(a3.evalf()), float(b3.evalf()), 42)
        v_param = np.linspace(0, 1, 42)
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
                x=X, y=Y, z=Z_high, colorscale="Tealgrn", opacity=surface_opacity,
                lighting=lighting_cfg, contours=contour_cfg, name="Upper Surface",
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z (top): %{z:.2f}<extra></extra>"
            )
        )
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z_low, colorscale="Ice", opacity=max(0.25, surface_opacity - 0.25),
                showscale=False, lighting=lighting_cfg, contours=contour_cfg, name="Lower Surface",
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z (bottom): %{z:.2f}<extra></extra>"
            )
        )

    elif "Cylindrical" in coord_sys and v_names == ["z", "r", "theta"]:
        th_vals = np.linspace(float(a3.evalf()), float(b3.evalf()), 56)
        p_vals = np.linspace(0, 1, 40)
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
                x=X, y=Y, z=Z_high, colorscale="Tealgrn", opacity=surface_opacity,
                lighting=lighting_cfg, contours=contour_cfg, name="Top Cap",
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z (top): %{z:.2f}<extra></extra>"
            )
        )
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z_low, colorscale="Ice", opacity=max(0.3, surface_opacity - 0.15),
                showscale=False, lighting=lighting_cfg, contours=contour_cfg, name="Bottom Base",
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z (bottom): %{z:.2f}<extra></extra>"
            )
        )

        X_outer = R_max_g * np.cos(TH)
        Y_outer = R_max_g * np.sin(TH)
        Z_wall_low = np.broadcast_to(f_z_min(R_max_g, TH), TH.shape)
        Z_wall_high = np.broadcast_to(f_z_max(R_max_g, TH), TH.shape)
        if np.any(np.abs(Z_wall_high - Z_wall_low) > 1e-6):
            Z_wall = Z_wall_low + P * (Z_wall_high - Z_wall_low)
            fig.add_trace(
                go.Surface(
                    x=X_outer, y=Y_outer, z=Z_wall, colorscale="Tealgrn", opacity=surface_opacity,
                    showscale=False, lighting=lighting_cfg, contours=contour_cfg, name="Curved Side Wall",
                    hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z (wall): %{z:.2f}<extra></extra>"
                )
            )

        if np.any(R_min_g > 1e-6):
            X_inner = R_min_g * np.cos(TH)
            Y_inner = R_min_g * np.sin(TH)
            Z_in_low = np.broadcast_to(f_z_min(R_min_g, TH), TH.shape)
            Z_in_high = np.broadcast_to(f_z_max(R_min_g, TH), TH.shape)
            Z_in_wall = Z_in_low + P * (Z_in_high - Z_in_low)

            fig.add_trace(
                go.Surface(
                    x=X_inner, y=Y_inner, z=Z_in_wall, colorscale="Ice", opacity=surface_opacity,
                    showscale=False, lighting=lighting_cfg, contours=contour_cfg, name="Inner Wall",
                    hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z (inner): %{z:.2f}<extra></extra>"
                )
            )

    elif "Cylindrical" in coord_sys and v_names == ["r", "theta", "z"]:
        z_min_val, z_max_val = float(a3.evalf()), float(b3.evalf())
        z_vals = np.linspace(z_min_val, z_max_val, 85 if is_starship else 55)
        p_vals = np.linspace(0, 1, 64)
        Z_grid, P = np.meshgrid(z_vals, p_vals)

        f_th_min = sp.lambdify(z, a2, "numpy")
        f_th_max = sp.lambdify(z, b2, "numpy")
        TH_min_g = np.broadcast_to(f_th_min(Z_grid), Z_grid.shape)
        TH_max_g = np.broadcast_to(f_th_max(Z_grid), Z_grid.shape)
        TH = TH_min_g + P * (TH_max_g - TH_min_g)

        f_r_min = sp.lambdify((theta, z), a1, "numpy")
        f_r_max = sp.lambdify((theta, z), b1, "numpy")
        R_in = np.broadcast_to(f_r_min(TH, Z_grid), Z_grid.shape)
        R_out = np.broadcast_to(f_r_max(TH, Z_grid), Z_grid.shape)

        hull_palette = [[0, "#94a3b8"], [0.5, "#cbd5e1"], [1, "#f8fafc"]] if is_starship else "Tealgrn"

        # 1. Outer Hull / Casing
        fig.add_trace(
            go.Surface(
                x=R_out * np.cos(TH), y=R_out * np.sin(TH), z=Z_grid,
                colorscale=hull_palette, opacity=surface_opacity,
                showscale=not is_starship,
                lighting=lighting_cfg, contours=contour_cfg, name="Outer Hull",
                hovertemplate="x: %{x:.2f} m<br>y: %{y:.2f} m<br>z: %{z:.2f} m<extra></extra>"
            )
        )
        # 2. Inner Wall (for hollow shells)
        if np.any(R_in > 1e-6):
            fig.add_trace(
                go.Surface(
                    x=R_in * np.cos(TH), y=R_in * np.sin(TH), z=Z_grid,
                    colorscale="Sunsetdark", opacity=surface_opacity, showscale=False,
                    lighting=lighting_cfg, contours=contour_cfg, name="Inner Wall",
                    hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>"
                )
            )
        # 3. End Caps
        th_1d = np.linspace(float(f_th_min(z_max_val)), float(f_th_max(z_max_val)), 56)
        rad_t = np.linspace(0, 1, 12)
        TH_c, RAD_t = np.meshgrid(th_1d, rad_t)
        for z_end in [z_min_val, z_max_val]:
            r_i_end = np.broadcast_to(f_r_min(TH_c, z_end), TH_c.shape)
            r_o_end = np.broadcast_to(f_r_max(TH_c, z_end), TH_c.shape)
            if np.any(np.abs(r_o_end - r_i_end) > 1e-4):
                R_cap = r_i_end + RAD_t * (r_o_end - r_i_end)
                fig.add_trace(
                    go.Surface(
                        x=R_cap * np.cos(TH_c), y=R_cap * np.sin(TH_c), z=np.full_like(R_cap, z_end),
                        colorscale="Ice", opacity=surface_opacity, showscale=False,
                        lighting=lighting_cfg, contours=contour_cfg, name=f"Cap (z={z_end})",
                        hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>"
                    )
                )

        # Attach detailed V3 aerodynamic & propulsion assembly if Starship preset is selected
        if is_starship:
            add_starship_v3_hardware(fig, surface_opacity)

    elif "Spherical" in coord_sys and v_names == ["rho", "phi", "theta"]:
        th_vals = np.linspace(float(a3.evalf()), float(b3.evalf()), 48)
        p_vals = np.linspace(0, 1, 36)
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
                x=X, y=Y, z=Z, colorscale="Tealgrn", opacity=surface_opacity,
                lighting=lighting_cfg, contours=contour_cfg, name="Spherical Shell",
                hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>"
            )
        )
    else:
        st.info("3D preview supports `z, y, x`, `z, r, theta`, `r, theta, z`, and `rho, phi, theta`. Step-by-step calculation below works for all orders!")

    if view_angle == "Top-Down (xy-Shadow)":
        camera_eye = dict(x=0.01, y=0.01, z=2.1)
    elif view_angle == "Side Elevation":
        camera_eye = dict(x=2.1, y=0.01, z=0.08)
    else:
        camera_eye = dict(x=1.65, y=1.65, z=0.65) if is_starship else dict(x=1.55, y=1.55, z=0.95)

    clean_axis = dict(
        showbackground=False,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zerolinecolor="rgba(56, 189, 248, 0.55)",
        zerolinewidth=2,
        showspikes=False,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(title="X (m)", **clean_axis),
            yaxis=dict(title="Y (m)", **clean_axis),
            zaxis=dict(title="Z (m)", **clean_axis),
            aspectmode="data",
            camera=dict(eye=camera_eye),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=640 if is_starship else 560,
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
        dec_latex = rf"\Huge \approx {float(final_vol.evalf()):.2f}\text{{ cubic units}}"
    except Exception:
        dec_latex = r"\Huge \text{Symbolic}"

    with st.container(border=True):
        f1, f2 = st.columns(2)
        with f1:
            st.markdown('<div class="final-tag">Final Exact Volume</div>', unsafe_allow_html=True)
            st.latex(rf"\Huge V = {sp.latex(final_vol)}")
        with f2:
            st.markdown('<div class="final-tag">Decimal Approximation</div>', unsafe_allow_html=True)
            st.latex(dec_latex)

except Exception as e:
    st.error(f"Invalid math input. Check variable names match your chosen coordinate system. Details: {e}")
                
