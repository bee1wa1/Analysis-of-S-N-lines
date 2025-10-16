# Analysis-of-S-N-lines
This Streamlit app lets you load fatigue-test datasets (CSV/TXT) and build normative S–N curves you can compare across materials, geometries, temperatures, and load ratios. Import is robust: you can choose encoding/delimiter, skip uncomplete rows, and there’s an optional debug expander to inspect raw headers. Columns for stress amplitude (σa) and cycles (N) are auto-detected; optional fields (Material, Geometry, Temperature, R, Valid?, Runout?) enable filtering.

Curves are fitted using the horizontal-distance regression prescribed by standards: in log–log space the model is log10 N = A + B · log10 σ. The app computes A, B, residual scatter s (on log N), the inverse slope k = −B, and the PS50 stress at 1e6 cycles σ@1e6 = 10^((6−A)/B).

Visualization is clear and consistent. For each selected group you’ll see:

- PS50 (solid line) and PS10/PS90 (dashed), all in the same color per group.

- An optional 95% confidence band around PS50 (translucent fill that doesn’t hide lines or points).

- Your test data points with matching marker color.

Axes are log-scaled in N and σa; the y-limit logic avoids pathological ranges on log scales. You can tune the number of points sampled along N for smooth curves.

The Parameters table summarizes A, B, k, σ@1e6, sample size, and data ranges for every displayed curve. Each row is color-highlighted to match the plot’s band color for immediate identification. One-click buttons export the parameter table as CSV; the plot is downloadable via the Plotly toolbar (or as PNG when server-side export is available).

Extras include an example dataset to get started instantly, legend grouping to reduce clutter, and consistent styling for multi-curve comparisons. In short: upload your data, filter, fit, visualize PS50/10/90 by group, and export the key design parameters you need.
