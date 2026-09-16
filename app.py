""" POD-FCDNN Streamlit Web Application
Interactive dashboard for POD-based surrogate modeling of fluid dynamics.
"""

import os
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import streamlit as st
from engine import load_checkpoint, predict_and_reconstruct

# Page Configuration
st.set_page_config(
    page_title="FluidPulse AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ FluidPulse AI")
st.markdown(
    """
Rapid CFD surrogate modeling powered by POD-FCDNN neural architectures. Real-time flow field prediction and spatial reconstruction.
"""
)

# SIDEBAR: CONTROLS & PARAMETERS
with st.sidebar:
    st.header("Control Panel")

    # CASE SELECTION
    case = st.selectbox(
        "Select Case", ["Cavity", "Cylinder", "Backward Facing Step", "NACA0012"]
    )

    # PARAMETER INPUT
    if case == "NACA0012":
        param = st.slider(
            "Angle of Attack (α)",
            min_value=-5.0,
            max_value=15.0,
            value=0.0,
            step=0.5,
        )
    else:
        param = st.slider(
            "Reynolds Number", min_value=100, max_value=10000, value=1000, step=100
        )

    # VARIABLE SELECTION
    selected_variable = st.radio(
        "Select Flow Variable",
        ["All Variables", "Absolute Pressure", "U Velocity", "V Velocity"],
        index=0,
    )

    # PREDICT BUTTON
    predict_btn = st.button("Predict Flow Field", use_container_width=True)


# LOAD CHECKPOINT WITH CACHING FOR MAXIMUM SPEED
@st.cache_resource
def get_model(case_name):
    checkpoint_filenames = {
        "Cavity": "cavity_checkpoint.pt",
        "Cylinder": "cylinder_checkpoint.pt",
        "Backward Facing Step": "bfs_checkpoint.pt",
        "NACA0012": "naca_checkpoint.pt",
    }

    filename = checkpoint_filenames[case_name]
    base_dir = os.path.dirname(os.path.abspath(__file__))

    possible_paths = [
        os.path.join(base_dir, "checkpoints", filename),
        os.path.join(base_dir, filename),
        filename,
    ]

    target_path = None
    for path in possible_paths:
        if os.path.exists(path):
            target_path = path
            break

    if not target_path:
        raise FileNotFoundError(
            f"Checkpoint file '{filename}' not found in 'checkpoints/' or repository root."
        )

    return load_checkpoint(target_path)


# CACHED KDTREE MESH INTERPOLATOR FOR ULTRA-FAST RENDER
@st.cache_data
def get_fast_grid_indices(x_coords, y_coords, x_min, x_max, y_min, y_max, res=180):
    grid_x_1d = np.linspace(x_min, x_max, res)
    grid_y_1d = np.linspace(y_min, y_max, res)
    grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)
    
    # Build spatial KD-Tree for instant lookup
    points = np.column_stack((x_coords, y_coords))
    tree = cKDTree(points)
    
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    _, indices = tree.query(grid_points)
    
    return grid_x_1d, grid_y_1d, indices, grid_x.shape


def generate_naca0012_path(c=1.0, alpha_deg=0.0, num_points=100):
    """Generates a Plotly SVG path string for a NACA 0012 airfoil rotated by Angle of Attack (alpha)."""
    x = np.linspace(0, c, num_points)
    yt = 5 * 0.12 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    x_coords = np.concatenate([x, x[::-1]])
    y_coords = np.concatenate([yt, -yt[::-1]])
    
    rad = np.radians(-alpha_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    x_rot = x_coords * cos_a - y_coords * sin_a
    y_rot = x_coords * sin_a + y_coords * cos_a

    path_str = f"M {x_rot[0]},{y_rot[0]}"
    for xv, yv in zip(x_rot[1:], y_rot[1:]):
        path_str += f" L {xv},{yv}"
    path_str += " Z"
    return path_str


# MAIN DASHBOARD AREA
if predict_btn:
    try:
        trainer = get_model(case)
        result = predict_and_reconstruct(trainer, param)

        u = result["u"]
        v = result["v"]
        p = result["p"]
        xy = result["xy"]

        x_coords = xy[:, 0]
        y_coords = xy[:, 1]

        if case == "Cylinder":
            x_min, x_max = -1.0, 5.0
            y_min, y_max = -1.5, 1.5
            use_fast_kdtree = False
        elif case == "Backward Facing Step":
            x_min, x_max = 0.00, 0.12
            y_min, y_max = -0.005, 0.005
            use_fast_kdtree = True  # Enable ultra-fast caching lookup
        elif case == "NACA0012":
            x_min, x_max = -0.5, 1.8
            y_min, y_max = -0.8, 0.8
            use_fast_kdtree = False
        else:
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            use_fast_kdtree = False

        if use_fast_kdtree:
            # INSTANT RE-INDEXING VIA KD-TREE
            grid_x_1d, grid_y_1d, indices, grid_shape = get_fast_grid_indices(
                x_coords, y_coords, x_min, x_max, y_min, y_max, res=180
            )
            grid_p = p[indices].reshape(grid_shape)
            grid_u = u[indices].reshape(grid_shape)
            grid_v = v[indices].reshape(grid_shape)
        else:
            # STANDARD CUBIC INTERPOLATION FOR ACCURATE CURVED GEOMETRIES
            grid_x_1d = np.linspace(x_min, x_max, 280)
            grid_y_1d = np.linspace(y_min, y_max, 280)
            grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)

            grid_p = griddata((x_coords, y_coords), p, (grid_x, grid_y), method="cubic")
            grid_u = griddata((x_coords, y_coords), u, (grid_x, grid_y), method="cubic")
            grid_v = griddata((x_coords, y_coords), v, (grid_x, grid_y), method="cubic")

            if case == "Cylinder":
                inside_cylinder = (grid_x**2 + grid_y**2) < (0.5**2)
                grid_p[inside_cylinder] = np.nan
                grid_u[inside_cylinder] = np.nan
                grid_v[inside_cylinder] = np.nan

        st.success(f"Prediction completed for {case}")

        def create_flow_figure(z_data, colorscale="Turbo", height=480):
            fig = go.Figure(
                data=go.Contour(
                    x=grid_x_1d,
                    y=grid_y_1d,
                    z=z_data,
                    colorscale=colorscale,
                    line_smoothing=1.1,
                    contours=dict(
                        coloring="heatmap",
                        showlines=False,
                    ),
                    line=dict(width=0),
                    colorbar=dict(
                        len=0.9,
                        thickness=14,
                        tickfont=dict(size=10),
                    ),
                )
            )

            shapes = []
            if case == "Cylinder":
                shapes.append(
                    dict(
                        type="circle",
                        xref="x",
                        yref="y",
                        x0=-0.5,
                        y0=-0.5,
                        x1=0.5,
                        y1=0.5,
                        fillcolor="black",
                        line=dict(color="red", width=1.5),
                    )
                )
            elif case == "NACA0012":
                shapes.append(
                    dict(
                        type="path",
                        path=generate_naca0012_path(alpha_deg=param),
                        fillcolor="black",
                        line=dict(color="red", width=1.5),
                    )
                )
            elif case == "Backward Facing Step":
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=-1.0,
                        y0=-0.5,
                        x1=0.0,
                        y1=0.0,
                        fillcolor="black",
                        line=dict(color="red", width=1.5),
                    )
                )

            fig.update_layout(
                xaxis=dict(
                    title="x",
                    range=[x_min, x_max],
                    showgrid=False,
                    zeroline=False,
                    constrain="domain",
                ),
                yaxis=dict(
                    title="y",
                    range=[y_min, y_max],
                    scaleanchor="x",
                    scaleratio=1,
                    showgrid=False,
                    zeroline=False,
                    constrain="domain",
                ),
                shapes=shapes,
                margin=dict(l=15, r=15, t=15, b=15),
                height=height,
            )
            return fig

        plotly_config = {"displayModeBar": False}

        if selected_variable == "All Variables":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Absolute Pressure")
                fig_p = create_flow_figure(grid_p, colorscale="Turbo", height=450)
                st.plotly_chart(fig_p, use_container_width=True, config=plotly_config)
            with col2:
                st.subheader("U Velocity")
                fig_u = create_flow_figure(grid_u, colorscale="Turbo", height=450)
                st.plotly_chart(fig_u, use_container_width=True, config=plotly_config)
            with col3:
                st.subheader("V Velocity")
                fig_v = create_flow_figure(grid_v, colorscale="Turbo", height=450)
                st.plotly_chart(fig_v, use_container_width=True, config=plotly_config)

        elif selected_variable == "Absolute Pressure":
            st.subheader("Absolute Pressure")
            fig_p = create_flow_figure(grid_p, colorscale="Turbo", height=650)
            st.plotly_chart(fig_p, use_container_width=True, config=plotly_config)

        elif selected_variable == "U Velocity":
            st.subheader("U Velocity")
            fig_u = create_flow_figure(grid_u, colorscale="Turbo", height=650)
            st.plotly_chart(fig_u, use_container_width=True, config=plotly_config)

        elif selected_variable == "V Velocity":
            st.subheader("V Velocity")
            fig_v = create_flow_figure(grid_v, colorscale="Turbo", height=650)
            st.plotly_chart(fig_v, use_container_width=True, config=plotly_config)

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
else:
    st.info(
        "Select parameters on the left sidebar and click **Predict Flow Field** to run the simulation."
    )
    
