import numpy as np
import matplotlib.pyplot as plt


benchmarks = [
    "SEEDBench",
    "MMVet",
    "VQAv2",
    "POPE",
    "GQA",
    "MME",
]


rows = [
    ["TokenFlow (13B)", 62.6, 27.7, 73.9, 85.0, 60.3, 1623],
    ["VILA-U (7B)", 59.0, 27.7, 75.3, 83.9, 58.3, 1336],
    ["UniTok (7B)", None, 33.9, 76.8, 83.2, 61.1, 1448],
    ["OpenVision 3 (7B)", 68.6, None, None, 86.6, 62.0, 1449],
    ["Ours (9B)", 70.0, 31.4, 77.4, 87.4, 63.5, 1489],
]


highlight_method = "Ours (9B)"
save_path = "radar_chart.png"
visual_floor = 0.22

axis_ranges = {
    "SEEDBench": (58.0, 72.0),
    "MMVet": (0.0, 40.0),
    "VQAv2": (72.0, 78.0),
    "POPE": (82.0, 88.0),
    "GQA": (56.0, 64.0),
    "MME": (0.0, 1800.0),
}

grid_levels = [
    visual_floor + 0.25 * (1.0 - visual_floor),
    visual_floor + 0.50 * (1.0 - visual_floor),
    visual_floor + 0.75 * (1.0 - visual_floor),
]


method_names = [r[0] for r in rows]
values = np.array(
    [
        [np.nan if v is None else float(v) for v in r[1:]]
        for r in rows
    ],
    dtype=float,
)

num_methods, num_benchmarks = values.shape
if num_benchmarks != len(benchmarks):
    raise ValueError(
        f"rows has {num_benchmarks} benchmark values, "
        f"but benchmarks contains {len(benchmarks)} names."
    )


axis_min = np.array([axis_ranges[b][0] for b in benchmarks], dtype=float)
axis_max = np.array([axis_ranges[b][1] for b in benchmarks], dtype=float)


def normalize(vals):
    norm = (vals - axis_min) / (axis_max - axis_min)
    return np.clip(norm, 0.0, 1.0)


def to_visual_radius(norm_vals):
    return visual_floor + norm_vals * (1.0 - visual_floor)


def interpolate_scale_value(radius, benchmark_idx):
    normalized_radius = (radius - visual_floor) / (1.0 - visual_floor)
    normalized_radius = np.clip(normalized_radius, 0.0, 1.0)
    return axis_min[benchmark_idx] + normalized_radius * (
        axis_max[benchmark_idx] - axis_min[benchmark_idx]
    )


def format_value(value, benchmark):
    if benchmark == "MME":
        return f"{int(round(value))}"
    return f"{value:.1f}"


def label_alignment(angle):
    cosine = np.cos(angle)
    sine = np.sin(angle)

    if cosine > 0.18:
        ha = "left"
    elif cosine < -0.18:
        ha = "right"
    else:
        ha = "center"

    if sine > 0.28:
        va = "bottom"
    elif sine < -0.28:
        va = "top"
    else:
        va = "center"

    return ha, va


norm_values = normalize(values)
visual_values = to_visual_radius(norm_values)

angles = np.linspace(0, 2 * np.pi, num_benchmarks, endpoint=False)
angles_closed = np.concatenate([angles, [angles[0]]])

fig = plt.figure(figsize=(9, 7), dpi=220)
ax = plt.subplot(111, polar=True)
ax.set_position([0.07, 0.10, 0.64, 0.78])

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_ylim(0, 1)
ax.set_yticks(grid_levels)
ax.set_yticklabels([])

ax.spines["polar"].set_visible(False)
ax.patch.set_visible(False)

ax.yaxis.grid(False)
ax.xaxis.grid(True, color="#999999", alpha=0.15, linewidth=0.6)
for r in grid_levels:
    ax.plot(
        angles_closed,
        [r] * len(angles_closed),
        color="#999999",
        linewidth=0.6,
        alpha=0.15,
        zorder=0,
    )

ax.set_xticks(angles)
ax.set_xticklabels(benchmarks, fontsize=12, fontweight="normal")
ax.tick_params(axis="x", pad=9)


default_colors = [
    "#ffb000",
    "#7db7ff",
    "#b0b000",
    "#ff7ab8",
    "#9c9c9c",
    "#6f73ff",
    "#00a676",
    "#d95f02",
    "#7570b3",
    "#66a61e",
]

color_map = {}
for i, name in enumerate(method_names):
    color_map[name] = default_colors[i % len(default_colors)]
color_map[highlight_method] = "#5f66ff"


for i, method in enumerate(method_names):
    vals_for_plot = visual_values[i].copy()
    valid_mask = ~np.isnan(values[i])

    vals_for_plot[~valid_mask] = visual_floor
    vals_closed = np.concatenate([vals_for_plot, [vals_for_plot[0]]])

    is_highlight = method == highlight_method
    linewidth = 1.15 if is_highlight else 1.0
    alpha = 0.86 if is_highlight else 0.76
    zorder = 4 if is_highlight else 2

    ax.plot(
        angles_closed,
        vals_closed,
        label=method,
        color=color_map[method],
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
    )

    if is_highlight:
        ax.fill(
            angles_closed,
            vals_closed,
            color=color_map[method],
            alpha=0.045,
            zorder=1,
        )


# Show the tick value for every grid layer on every axis.
for j, benchmark in enumerate(benchmarks):
    ha, va = label_alignment(angles[j])
    for radius in grid_levels:
        ax.text(
            angles[j],
            min(radius + 0.012, 0.98),
            format_value(interpolate_scale_value(radius, j), benchmark),
            fontsize=7.6,
            color="#666666",
            alpha=0.70,
            ha=ha,
            va=va,
            zorder=1,
            bbox=dict(
                boxstyle="round,pad=0.07",
                facecolor="white",
                edgecolor="none",
                alpha=0.46,
            ),
        )


ax.legend(
    loc="center left",
    bbox_to_anchor=(0.98, 0.48),
    fontsize=9,
    frameon=False,
    handlelength=1.7,
    labelspacing=0.5,
)

plt.savefig(save_path, bbox_inches="tight", pad_inches=0.25)
plt.show()

print(f"Saved to: {save_path}")
