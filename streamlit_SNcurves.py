import io
import math
import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm, t

st.set_page_config(page_title="S–N Curves • PS50/10/90 (95% conf.)", layout="wide")

palette = px.colors.qualitative.Plotly

# ---------------------- Utils ----------------------
Z10 = norm.ppf(0.10)   # -1.28155...
Z90 = norm.ppf(0.90)   # +1.28155...

REQ_COLS = {
    "σ": ["σn,a [MPa]", "sigma_a", "sigma", "Stress", "Sigma_a", "sigma_a_MPa"],
    "N": ["N", "Cycles", "n_cycles"],
    "Materiale": ["Materiale", "Material", "Mat"],
    "Geometria": ["Geometria", "Geometry", "Geom"],
    "Temperatura": ["Temperatura", "Temperature", "Temp"],
    "R": ["R", "R-ratio"],
    "ProvaValida": ["Prova valida?", "Valid?", "Valid"],
    "Runout": ["Runout?", "Runout"],
}

def find_col(df, keys):
    for k in keys:
        if k in df.columns:
            return k
    return None

def to_numeric_safe(s):
    return pd.to_numeric(
        s.astype(str)
         .str.replace(",", ".", regex=False)
         .str.replace("−", "-", regex=False)
         .str.replace(" ", "", regex=False),
        errors="coerce"
    )

def horizontal_regression(x_logsig, y_logn):
    """Normative horizontal-distance fit (closed form)."""
    x = np.asarray(x_logsig, float)
    y = np.asarray(y_logn, float)
    n = x.size
    SX, SY = x.sum(), y.sum()
    SXX, SXY = np.dot(x, x), np.dot(x, y)

    denom = (n * SXY - SX * SY)
    if denom == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    m_horiz_b = (n * SXX - SX * SX) / denom
    B = 1.0 / m_horiz_b
    Xbar, Ybar = SX / n, SY / n
    A = Ybar - B * Xbar

    yhat = A + B * x
    s = np.sqrt(np.sum((y - yhat) ** 2) / max(n - 2, 1))
    Sxx = np.sum((x - Xbar) ** 2)
    return A, B, s, Xbar, Sxx

def sigma_from_N(A, B, s_shift, logN):
    # logσ = (logN - A - s_shift) / B
    return 10 ** ((logN - A - s_shift) / B)

# safe PNG export (doesn't crash if Kaleido/Chrome missing)
def encode_png(fig):
    try:
        import plotly.io as pio
        png = pio.to_image(fig, format="png", scale=2)
        b64 = base64.b64encode(png).decode()
        return f"data:file/png;base64,{b64}"
    except Exception:
        return None

def with_alpha(hex_color, alpha=0.18):
    """Convert '#RRGGBB' to 'rgba(r,g,b,a)'."""
    c = hex_color.replace("#", "")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ---------------------- UI ----------------------
st.title("S–N Curves • PS50 / PS10 / PS90 (95% confidence)")

st.markdown(
    "Upload a table containing at least **σa [MPa]** (stress amplitude) and **N** (cycles). "
    "For comparisons, include **Material**, **Geometry**, **Temperature**, **R**. "
    "Optional filters: *Valid?* and *Runout?*."
)

file = st.file_uploader("Upload file (CSV / XLSX / TXT)", type=["csv", "xlsx", "xls", "txt"])

# Example dataset
example = pd.DataFrame({
    "Material": ["Material A"]*5,
    "Geometry": ["Plain"]*5,
    "Temperature (°C)": ["23","23","23","23","23"],
    "R": [0.05, 0.05, 0.05, 0.05, 0.05],
    "Valid test?": ["Yes","Yes","Yes","Yes","Yes"],
    "σn,a [MPa]": [6.3, 5.5, 5.1, 4.3, 4.2],
    "N": [2.08e6, 3.5e6, 6.1e6, 1.2e7, 2.3e7],
    "Runout?": ["No","No","No","No","Yes"]
})
exp = st.expander("Show / load example dataset")
with exp:
    st.dataframe(example, use_container_width=True)
    if st.button("Use example"):
        file = io.BytesIO()
        example.to_csv(file, index=False)
        file.seek(0)

# ---------------------- Load & normalize ----------------------
if file is not None:
    # normalize bytes and name (works for UploadedFile and BytesIO)
    if isinstance(file, io.BytesIO):
        file_bytes = file.getvalue()
        file_name = "uploaded.csv"
    else:
        file_bytes = file.read()
        file_name = getattr(file, "name", "uploaded.csv")

    with st.sidebar:
        st.header("File reading")
        enc_opt = st.selectbox("Encoding", ["Auto", "utf-8", "utf-8-sig", "latin-1", "cp1252"], index=0)
        sep_opt = st.selectbox("Delimiter", ["Auto", ",", ";", "\\t", "|"], index=0)
        allow_bad = st.checkbox("Skip incomplete rows", value=True)
        no_header = st.checkbox("No header row (header=None)", value=False)
        curve_points = st.slider("Curve resolution (points on N)", 20, 400, 100, step=10)

    def try_read_csv(b, sep, enc, allow_bad=False, header_infer=True):
        return pd.read_csv(
            io.BytesIO(b),
            sep=sep,
            engine="python",
            encoding=None if enc is None else enc,
            encoding_errors="replace",
            quotechar='"',
            escapechar='\\',
            skip_blank_lines=True,
            on_bad_lines="skip" if allow_bad else "error",
            header=None if not header_infer else "infer"
        )

    # Excel
    if file_name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        # Text/CSV with robust autodetect
        if sep_opt == "Auto":
            seps_to_try = [None, ";", ",", "\t", "|", r"\s*;\s*", r"\s*,\s*"]
        else:
            seps_to_try = ["\t"] if sep_opt == "\\t" else [sep_opt]
        encs_to_try = [None, "utf-8-sig", "utf-8", "latin-1", "cp1252"] if enc_opt == "Auto" else [enc_opt]

        df, last_err = None, None
        for enc in encs_to_try:
            for sep in seps_to_try:
                try:
                    df = try_read_csv(file_bytes, sep, enc, allow_bad=allow_bad, header_infer=not no_header)
                    if df.shape[1] == 1 and (sep is None or (isinstance(sep, str) and sep.startswith(r"\s"))):
                        df = None
                        continue
                    break
                except Exception as e:
                    last_err = e
                    df = None
            if df is not None:
                break
        if df is None:
            st.error(f"Unable to read file. Last error:\n{last_err}")
            st.stop()

    # ---- Column mapping ----
    col_sigma = find_col(df, REQ_COLS["σ"])
    col_N     = find_col(df, REQ_COLS["N"])
    col_mat   = find_col(df, REQ_COLS["Materiale"]) or "Materiale"
    col_geo   = find_col(df, REQ_COLS["Geometria"]) or "Geometria"
    col_temp  = find_col(df, REQ_COLS["Temperatura"]) or "Temperatura"
    col_R     = find_col(df, REQ_COLS["R"]) or "R"
    col_valid = find_col(df, REQ_COLS["ProvaValida"])
    col_runout= find_col(df, REQ_COLS["Runout"])

    with st.expander("🧭 Column mapping (detected)"):
        st.write("σa column:", col_sigma, "| exists:", col_sigma in df.columns if col_sigma else False)
        st.write("N column:", col_N, "| exists:", col_N in df.columns if col_N else False)
        st.write("All columns:", list(df.columns))

    # Ensure group-by fields exist
    for c in [col_mat, col_geo, col_temp, col_R]:
        if c not in df.columns:
            df[c] = "—"

    # Numerics
    if col_sigma is None or col_N is None:
        st.error("Could not detect σa and/or N columns. Please adjust your header names.")
        st.stop()
    df[col_sigma] = to_numeric_safe(df[col_sigma])
    df[col_N]     = to_numeric_safe(df[col_N])

    # ---------------------- Filters ----------------------
    with st.sidebar:
        st.header("Filters")
        mats  = st.multiselect("Material", sorted(df[col_mat].dropna().unique().tolist()))
        geos  = st.multiselect("Geometry", sorted(df[col_geo].dropna().unique().tolist()))
        temps = st.multiselect("Temperature", sorted(df[col_temp].dropna().unique().tolist()))
        Rs    = st.multiselect("R-ratio", sorted(df[col_R].dropna().unique().tolist()))
        only_valid   = st.checkbox("Only **valid** tests", value=True if col_valid else False)
        exclude_run  = st.checkbox("Exclude **runouts**", value=True if col_runout else False)
        grp_cols = st.multiselect(
            "Group curves by…",
            [col_mat, col_geo, col_temp, col_R],
            default=[col_mat, col_geo, col_temp, col_R]
        )
        show_CI = st.checkbox("Show 95% confidence band (PS50)", value=True)

    sel = pd.Series(True, index=df.index)
    if mats:  sel &= df[col_mat].isin(mats)
    if geos:  sel &= df[col_geo].isin(geos)
    if temps: sel &= df[col_temp].isin(temps)
    if Rs:    sel &= df[col_R].isin(Rs)
    if only_valid and col_valid:
        sel &= df[col_valid].astype(str).str.lower().isin(["si","sì","yes","true","1","ja"])
    if exclude_run and col_runout:
        sel &= ~df[col_runout].astype(str).str.lower().isin(["si","sì","yes","true","1","ja"])

    data = df.loc[sel].copy()
    data = data[(data[col_sigma] > 0) & (data[col_N] > 0)].copy()
    if data.empty:
        st.warning("No data after filters.")
        st.stop()

    # ------------- Fit & plot -------------
    data["logN"] = np.log10(data[col_N])
    data["logσ"] = np.log10(data[col_sigma])

    if not grp_cols:
        grp_cols = [col_mat, col_geo, col_temp, col_R]
    data["CurveKey"] = data[grp_cols].astype(str).agg(" | ".join, axis=1)

    keys = sorted(data["CurveKey"].unique().tolist())
    sel_keys = st.multiselect("Select curves to compare", keys, default=keys)

    N_min_log = data["logN"].min()*0.98
    N_max_log = data["logN"].max()*1.02
    grid_logN = np.linspace(N_min_log, N_max_log, int(curve_points))
    grid_N = 10**grid_logN

    fig = go.Figure()
    rows = []

    for i, key in enumerate(sel_keys):
        color = palette[i % len(palette)]
        g = data[data["CurveKey"] == key]
        if len(g) < 2:
            continue

        A, B, s, Xbar, Sxx = horizontal_regression(g["logσ"], g["logN"])
        if np.isnan(A) or np.isnan(B):
            continue

        n = len(g)
        dof = max(n - 2, 1)
        tcrit = t.ppf(0.975, dof)

        # PS50/10/90 as σ(N)
        sig50 = sigma_from_N(A, B, 0.0, grid_logN)
        sig10 = sigma_from_N(A, B, Z10 * s, grid_logN)
        sig90 = sigma_from_N(A, B, Z90 * s, grid_logN)

        # 95% confidence band around PS50 (drawn first, translucent fill)
        def ci_band(slog):
            X = np.log10(slog)
            mult = tcrit * s * np.sqrt(1 + 1/n + ((X - Xbar) ** 2) / max(Sxx, 1e-12))
            upper = 10 ** ((grid_logN - A - mult) / B)
            lower = 10 ** ((grid_logN - A + mult) / B)
            return lower, upper

        if show_CI:
            lo50, up50 = ci_band(sig50)
            fig.add_trace(go.Scatter(
                x=grid_N, y=lo50, mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
                legendgroup=key
            ))
            fig.add_trace(go.Scatter(
                x=grid_N, y=up50, mode="lines", fill="tonexty",
                fillcolor=with_alpha(color, 0.18), line=dict(width=0),
                showlegend=False, hoverinfo="skip", legendgroup=key
            ))

        # Data points
        fig.add_trace(go.Scatter(
            x=g[col_N], y=g[col_sigma],
            mode="markers", name=f"{key} • data",
            marker=dict(size=12, color=color, line=dict(color="black", width=2)),
            hovertemplate="N=%{x:.3g}<br>σ=%{y:.3g} MPa",
            legendgroup=key, showlegend=False
        ))

        # PS50 solid, PS10/PS90 dashed — same color
        fig.add_trace(go.Scatter(
            x=grid_N, y=sig50, mode="lines",
            name=f"{key} • PS50%", line=dict(color=color, width=3),
            legendgroup=key
        ))
        fig.add_trace(go.Scatter(
            x=grid_N, y=sig10, mode="lines",
            name=f"{key} • PS10%", line=dict(color=color, dash="dash", width=2),
            legendgroup=key, showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=grid_N, y=sig90, mode="lines",
            name=f"{key} • PS90%", line=dict(color=color, dash="dash", width=2),
            legendgroup=key, showlegend=False
        ))

        # Parameters (include inverse slope and stress at 1e6 cycles)
        k_inverse = -B                      # positive inverse slope
        sigma_1e6 = 10 ** ((6.0 - A) / B)   # PS50 @ 1e6 cycles
        rows.append({
            "Curve": key, "n": n,
            "A": A, "B": B,
            "k (inverse slope)": k_inverse,
            "σ@1e6 [MPa] (PS50)": sigma_1e6,
            "s_logN": s,
            "N range": f"{g[col_N].min():.3g} – {g[col_N].max():.3g}",
            "σ range [MPa]": f"{g[col_sigma].min():.3g} – {g[col_sigma].max():.3g}",
            "Color": color,  # <-- NEW
        })

        # --- Y-axis limits from σ column ---
        min_sigma = float(data[col_sigma].min())
        max_sigma = float(data[col_sigma].max())

        ABS_PAD = 10.0  # your requested ±10 MPa
        REL_PAD = 0.20  # 20% fallback when ABS_PAD would push ≤ 0
        FLOOR = 1e-3  # MPa floor for log axis (tune if needed)

        lower_abs = min_sigma - ABS_PAD
        upper_abs = max_sigma + ABS_PAD

        # if absolute padding makes lower bound non-positive, fall back to relative padding
        if lower_abs <= 0:
            y_min = max(min_sigma * (1 - REL_PAD), FLOOR)
        else:
            y_min = lower_abs

        y_max = max(upper_abs, y_min * 1.05)  # ensure y_max > y_min

        fig.update_layout(
            xaxis=dict(type="log", title="N (cycles) — log scale", exponentformat="e"),
            yaxis=dict(type="log", title="σa (MPa)",
                       range=[np.log10(y_min), np.log10(y_max)])
        )

    # Axes & layout
    fig.update_layout(
        xaxis=dict(type="log", title="N (cycles) — log scale", exponentformat="e"),
        yaxis=dict(type="log", title="σa (MPa)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=30, r=10, t=30, b=30),
        height=650
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------------------- Parameters table ----------------------
    if rows:
        params = pd.DataFrame(rows)


        # Style rows using the curve color (light alpha, same as band)
        def _row_bg(r):
            c = r.get("Color", None)
            bg = with_alpha(c, 0.18) if isinstance(c, str) else "transparent"
            return [f"background-color: {bg}" for _ in r]


        styled = params.style.apply(_row_bg, axis=1)
        # Hide helper column if supported by your pandas version
        try:
            styled = styled.hide(axis="columns", subset=["Color"])
        except Exception:
            pass

        st.subheader("Estimated parameters (by selected curve)")
        st.dataframe(styled, use_container_width=True)

        csv = params.drop(columns=["Color"], errors="ignore").to_csv(index=False).encode()
        st.download_button("Download parameters (CSV)", csv, "sn_curve_parameters.csv", "text/csv")

        png_uri = encode_png(fig)
        if png_uri:
            st.markdown(f"[Download plot (PNG)]({png_uri})", unsafe_allow_html=True)
        else:
            st.info("Use the camera icon on the Plotly toolbar to download the image (client-side).")
    else:
        st.info("Select at least one curve with ≥2 valid points (σ>0 and N>0).")
