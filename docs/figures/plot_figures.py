# The figures of OpenSurfaces' documentation, from the CSV files of make_data.jl, in the light and dark renderings
# of the site (docs/src/assets/<name>-light.png, -dark.png):
#   for m in light dark; do FIGSTYLE_DOCS=$m ~/miniconda3/bin/python docs/figures/plot_figures.py; done
# (from the package root). Mesh panels are colored by their class (figstyle's MESH roles), cases and methods by the
# qualitative palette, fields by FIELD; exact values and error = tol are dashed lines in the ink color.
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Patch
from matplotlib.ticker import NullFormatter, LogLocator

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
fs.apply()


def read(name):
    """A CSV of make_data.jl as a dict of columns (numbers where they parse)."""
    with open(os.path.join(DATA, name), encoding="utf-8", newline="") as f:
        head, *rows = [r for r in csv.reader(f) if r]
    cols = {}
    for j, h in enumerate(head):
        col = [r[j] for r in rows]
        try:
            cols[h] = np.array([float(x) for x in col])
        except ValueError:
            cols[h] = np.array(col)
    return cols


def series(d, key):
    """The distinct values of a column, in the order of first appearance."""
    _, idx = np.unique(d[key], return_index=True)
    return [d[key][i] for i in sorted(idx)]


def loglog(ax):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())


def logy(ax, step=None):
    """A log y axis with a major tick every `step` decades (default: about five ticks)."""
    ax.set_yscale("log")
    if step is None:
        lo, hi = ax.get_ylim()
        step = max(1, int(np.ceil(np.log10(hi / lo) / 5)))
    ax.yaxis.set_major_locator(LogLocator(base=10 ** step, numticks=20))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(1.0,), numticks=40) if step > 1 else
                               LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=40))
    ax.yaxis.set_minor_formatter(NullFormatter())


def below(ax, ncol=1, **kw):
    """The legend under the axes, clear of the data."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), fontsize=7.5, ncol=ncol, **kw)


def bare(ax):
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---- gallery --------------------------------------------------------------------------------------------------------
def draw_mesh(ax, panels, curves, lw=0.3, alpha=0.22):
    ids = panels["panel"]
    starts = np.r_[0, np.nonzero(np.diff(ids))[0] + 1, len(ids)]
    for a, b in zip(starts[:-1], starts[1:]):
        c = fs.MESH[panels["class"][a]]
        xy = np.c_[panels["x"][a:b], panels["y"][a:b]]
        ax.add_patch(Polygon(xy, closed=True, facecolor=c, alpha=alpha, lw=0))
        ax.add_patch(Polygon(xy, closed=True, facecolor="none", edgecolor=c, lw=lw))
    draw_curves(ax, curves)


def draw_curves(ax, curves, lw=0.9):
    for cid in np.unique(curves["curve"]):
        m = curves["curve"] == cid
        x, y = curves["x"][m], curves["y"][m]
        ax.plot(np.r_[x, x[0]], np.r_[y, y[0]], color=fs.INK, lw=lw, solid_capstyle="round")


def gallery():
    panels = read("gallery_panels.csv"); curves = read("gallery_curves.csv"); fld = read("gallery_field.csv")
    xs = np.unique(fld["x"]); ys = np.unique(fld["y"])
    U = fld["u"].reshape(len(ys), len(xs))
    fig = plt.figure(figsize=(fs.WIDTH * 1.45, 2.05))
    gs = fig.add_gridspec(1, 4, width_ratios=[2.33, 1.05, 2.33, 0.07], wspace=0.08)
    ax0, ax1, ax2, cax = (fig.add_subplot(gs[0, i]) for i in range(4))
    lim = (-1.66, 3.26), (-1.06, 1.06)
    # (a) the mesh
    draw_mesh(ax0, panels, curves)
    ax0.set_xlim(*lim[0]); ax0.set_ylim(*lim[1]); bare(ax0)
    zx, zy = (1.44, 1.615), (-0.0875, 0.0875)            # the zoom box, at the tip of the outer ellipse
    ax0.add_patch(Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fill=False, ec=fs.INK, lw=0.8, zorder=5))
    handles = [Patch(facecolor=fs.MESH[k], edgecolor=fs.MESH[k], alpha=0.6, label=l) for k, l in
               (("weighted", "weighted edge layer"), ("graded", "graded layers"), ("interior", "interior"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.075), ncol=3, fontsize=8.5,
               handlelength=1.2, columnspacing=2.0)
    ax0.set_xlabel("film with two holes and a second drop")
    fs.panel(ax0, "a")
    # (b) the edge layers at Γ
    draw_mesh(ax1, panels, curves, lw=0.7, alpha=0.35)
    nodes = read("gallery_nodes.csv")
    for k in ("interior", "graded", "weighted"):
        m = nodes["class"] == k
        ax1.plot(nodes["x"][m], nodes["y"][m], "o", color=fs.MESH[k], ms=1.3, mew=0, zorder=4)
    ax1.set_xlim(*zx); ax1.set_ylim(*zy); ax1.set_aspect("equal"); ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_xlabel(r"edge layers at $\Gamma$")
    fs.panel(ax1, "b")
    # (c) the single layer potential of a unit density
    im = ax2.pcolormesh(xs, ys, np.ma.masked_invalid(U), cmap=fs.FIELD, shading="auto", rasterized=True)
    draw_curves(ax2, curves, lw=0.7)
    ax2.set_xlim(*lim[0]); ax2.set_ylim(*lim[1]); bare(ax2)
    ax2.set_xlabel(r"single layer potential, $S_\Omega[1]$")
    fs.panel(ax2, "c")
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_edgecolor(fs.INK); cb.ax.tick_params(labelsize=8, color=fs.INK, direction="in")
    pos = cax.get_position(); cax.set_position([pos.x0, pos.y0 + 0.12 * pos.height, pos.width, 0.76 * pos.height])
    fs.save(fig, "gallery")


# ---- convergence ----------------------------------------------------------------------------------------------------
LABELS = {
    "S[1] = E(r²)/π": r"$S[1] = E(r^2)/\pi$",
    "S[1/ω] = π/4": r"$S[1/\omega] = \pi/4$",
    "S[1/ω] = π/4 on Γ": r"$S[1/\omega] = \pi/4$ on $\Gamma$",
    "Stokeslet, rigid disc": r"Stokeslet, $S[e_x/2\pi\omega] = 3e_x/32$",
    "Laplace, near Γ": r"Laplace $\cos 5\theta$, near $\Gamma$",
    "Laplace, on Γ": r"Laplace $\cos 5\theta$, on $\Gamma$",
    "Stokes, near Γ": r"Stokes $S[n] = 0$, near $\Gamma$",
    "Stokes, on Γ": r"Stokes $S[n] = 0$, on $\Gamma$",
    "scalar, e^{−ck}/k": r"scalar, $e^{-ck}/k$",
    "tensor, (0, e^{−ck}/k)": r"tensor, $(0, e^{-ck}/k)$",
    "screened, e^{−κρ}/(4πρ)": r"screened, $e^{-\kappa\rho}/4\pi\rho$",
}


def by_case(ax, d, x, key, order=None):
    for k, name in enumerate(order or series(d, key)):
        m = (d[key] == name) & (d["error"] > 0)
        fs.line(ax, d[x][m], d["error"][m], fs.QUAL[k], label=LABELS.get(name, name))


def convergence():
    fig, axs = plt.subplots(1, 4, figsize=(fs.WIDTH * 1.35, 3.0))
    d = read("conv_area.csv")
    by_case(axs[0], d, "p", "case")
    logy(axs[0]); axs[0].set_xticks([6, 8, 10, 12, 14])
    axs[0].set_xlabel(r"order, $p$"); axs[0].set_ylabel("max error, area quadrature")
    below(axs[0]); fs.panel(axs[0], "a")
    d = read("conv_line.csv")
    by_case(axs[1], d, "p", "case", ["Laplace, on Γ", "Laplace, near Γ", "Stokes, on Γ", "Stokes, near Γ"])
    logy(axs[1]); axs[1].set_xticks([4, 8, 12, 16])
    axs[1].set_xlabel(r"order, $p$"); axs[1].set_ylabel("max error, boundary quadrature")
    below(axs[1]); fs.panel(axs[1], "b")
    d = read("conv_fast.csv")
    by_case(axs[2], d, "tol", "method")
    t = np.array([d["tol"].min(), d["tol"].max()])
    fs.ref(axs[2], t, t, label=r"error $=$ tol")
    loglog(axs[2]); axs[2].invert_xaxis()
    axs[2].set_xlabel(r"requested tolerance, tol"); axs[2].set_ylabel("relative error, fast sums")
    below(axs[2]); fs.panel(axs[2], "c")
    d = read("conv_symbol.csv")
    by_case(axs[3], d, "nq", "kernel")
    logy(axs[3]); axs[3].set_xticks([4, 8, 12, 16, 20]); axs[3].set_xlabel(r"Chebyshev points per panel, $n_q$")
    axs[3].set_ylabel("relative error, symbol tables")
    below(axs[3]); fs.panel(axs[3], "d")
    fig.tight_layout(w_pad=1.2)
    fs.save(fig, "convergence")


# ---- examples -------------------------------------------------------------------------------------------------------
def thin(r, n=70):
    """Indices of about n nodes spread evenly in r (the first node of each bin)."""
    edges = np.linspace(r.min(), r.max() + 1e-12, n + 1)
    k = np.digitize(r, edges)
    return np.array([np.nonzero(k == i)[0][0] for i in np.unique(k)])


def examples():
    fig, axs = plt.subplots(1, 4, figsize=(fs.WIDTH * 1.35, 3.0))
    d = read("ex_disc_density.csv")
    r = d["r"]; o = np.argsort(r); r, d = r[o], {k: v[o] for k, v in d.items()}
    rr = np.linspace(0, r.max(), 400)
    i = thin(r)
    # (a) the solved densities along a radius
    axs[0].plot(r[i], d["sigma"][i], "o", color=fs.QUAL[0], ms=2.6, label=r"charged disc, $\sigma$")
    axs[0].plot(r[i], d["fx"][i], "o", color=fs.QUAL[1], ms=2.6, label=r"rigid disc, $f_x$")
    fs.ref(axs[0], rr, 4 / (np.pi * np.sqrt(1 - rr ** 2)), label="exact")
    fs.ref(axs[0], rr, 8 / (3 * np.pi * np.sqrt(1 - rr ** 2)))
    logy(axs[0]); axs[0].set_xlabel(r"radius, $r$"); axs[0].set_ylabel("density at the nodes")
    below(axs[0]); fs.panel(axs[0], "a")
    # (b) its pointwise error against the distance to the edge
    axs[1].plot(1 - r, d["sigma_error"], "o", color=fs.QUAL[0], ms=1.4, mew=0, alpha=0.7, label=r"charged disc, $\sigma$")
    axs[1].plot(1 - r, d["f_error"], "o", color=fs.QUAL[1], ms=1.4, mew=0, alpha=0.7, label=r"rigid disc, $f_x$")
    loglog(axs[1]); axs[1].set_xlabel(r"distance to the edge, $1 - r$"); axs[1].set_ylabel("relative error at the nodes")
    below(axs[1], markerscale=2.5); fs.panel(axs[1], "b")
    # (c) convergence in p of the densities and the totals (charge, drag)
    d = read("ex_disc_conv.csv")
    for k, case in enumerate(["charged disc", "rigid disc"]):
        for q, style, lab in (("density", "-o", "density"), ("total", ":o", "charge $Q$" if k == 0 else "drag $F$")):
            m = (d["case"] == case) & (d["quantity"] == q)
            c = fs.QUAL[k]      # rigid disc: open squares (its totals sit on the charged disc's)
            axs[2].plot(d["p"][m], d["error"][m], style.replace("o", "o" if k == 0 else "s"), color=c,
                        mfc=c if k == 0 else "none", mec=c, ms=3.5 if k == 0 else 5, label=f"{case}, {lab}")
    logy(axs[2]); axs[2].set_xticks([8, 10, 12])
    axs[2].set_xlabel(r"order, $p$"); axs[2].set_ylabel("max relative error")
    below(axs[2]); fs.panel(axs[2], "c")
    # (d) the screened kernel's remainder table against its closed form
    d = read("ex_screened.csv")
    m = d["rho"] > 0
    e = np.abs(d["table"] - d["exact"]) / np.abs(d["exact"][0])
    fs.line(axs[3], d["rho"][m], e[m], fs.QUAL[0], marker=False, label=r"screened kernel, $\kappa = 4$")
    loglog(axs[3]); axs[3].set_xlabel(r"distance, $\rho$")
    axs[3].set_ylabel(r"table error, $|\tilde\ell - \ell|/|\ell(0)|$")
    below(axs[3]); fs.panel(axs[3], "d")
    fig.tight_layout(w_pad=1.2)
    fs.save(fig, "examples")


# ---- pointwise errors: the single layer of the elliptic disc's charge density ---------------------------------------
def binned_max(x, y, edges):
    """The largest y in each bin of x (bins without data dropped), at the bins' geometric centers."""
    k = np.digitize(x, edges)
    c = np.sqrt(edges[:-1] * edges[1:])
    out = [(c[i - 1], y[k == i].max()) for i in range(1, len(edges)) if np.any(k == i)]
    return np.array(out).T


def pointwise():
    fig, axs = plt.subplots(1, 3, figsize=(fs.WIDTH * 1.35, 3.1), gridspec_kw=dict(width_ratios=[1.55, 1, 1]))
    ax0, ax1, ax2 = axs
    # (a) the error on a grid of points of Ω, p = 10, tol = 1e-10 (far below the floor set by p)
    d = read("pointwise_map.csv")
    xs, ys = np.unique(d["x"]), np.unique(d["y"])
    E = d["error"].reshape(len(ys), len(xs))
    # the largest error within ±3 grid cells: the signed error changes sign inside the panels, which on a log scale
    # draws thin dark lines along its zeros; the envelope keeps the magnitude
    w = 3
    P = np.pad(np.where(np.isnan(E), -np.inf, E), w, constant_values=-np.inf)
    win = np.lib.stride_tricks.sliding_window_view(P, (2 * w + 1, 2 * w + 1)).max(axis=(2, 3))
    E = np.where(np.isnan(E), np.nan, win)
    lo = 1e-13
    im = ax0.pcolormesh(xs, ys, np.ma.masked_invalid(np.log10(np.maximum(E, lo))), cmap=fs.FIELD, shading="auto",
                        vmin=np.log10(lo), vmax=np.log10(3e-8), rasterized=True)
    pn = read("pointwise_panels.csv")
    for k in np.unique(pn["panel"]):
        m = pn["panel"] == k
        ax0.plot(pn["x"][m], pn["y"][m], color=fs.INK, lw=0.25, alpha=0.35)
    th = np.linspace(0, 2 * np.pi, 400)
    ax0.plot(xs.max() * np.cos(th), ys.max() * np.sin(th), color=fs.INK, lw=0.8)
    bare(ax0); fs.panel(ax0, "a")
    cb = fig.colorbar(im, ax=ax0, orientation="horizontal", fraction=0.07, pad=0.05, aspect=30)
    cb.set_ticks([-13, -12, -11, -10, -9, -8])
    cb.set_ticklabels([rf"$10^{{{k}}}$" for k in range(-13, -7)])
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label(r"$|u - V|/V$ on a grid, local max, $p = 10$, tol $= 10^{-10}$", fontsize=9)
    # (b) the largest error against the distance to the edge, p = 10, for four tolerances
    d = read("pointwise_dist.csv")
    tols = sorted(set(d["tol"]), reverse=True)
    edges = np.logspace(np.log10(max(d["d"].min(), 1e-4)) - 0.01, 0.001, 26)
    for c, t in zip(fs.seq(len(tols)), tols):
        m = d["tol"] == t
        x, y = binned_max(d["d"][m], np.maximum(d["error"][m], 1e-16), edges)
        fs.line(ax1, x, y, c, ms=2.5, label=rf"tol $= 10^{{{int(round(np.log10(t)))}}}$")
    loglog(ax1); ax1.set_xlabel(r"distance to the edge, $1 - \rho_e$")
    ax1.set_ylabel(r"largest error at the area nodes")
    below(ax1, ncol=2); fs.panel(ax1, "b")
    # (c) the largest error against tol, by p: area nodes (filled) and nodes of Γ (open)
    d = read("pointwise_tol.csv")
    ps = sorted(set(d["p"]))
    for c, p in zip(fs.seq(len(ps)), ps):
        for st, mk, mfc, lab in (("area nodes", "-o", c, "area"), ("nodes of Γ", "--s", "none", r"$\Gamma$")):
            m = (d["p"] == p) & (d["set"] == st)
            ax2.plot(d["tol"][m], d["error"][m], mk, color=c, mfc=mfc, mec=c, ms=3.5, lw=1,
                     label=rf"$p = {int(p)}$, {lab}")
    t = np.array([d["tol"].min(), d["tol"].max()])
    fs.ref(ax2, t, t, label=r"error $=$ tol")
    loglog(ax2); ax2.invert_xaxis()
    ax2.set_xlabel("requested tolerance, tol"); ax2.set_ylabel("largest relative error")
    below(ax2, ncol=2); fs.panel(ax2, "c")
    fig.tight_layout(w_pad=1.5)
    fs.save(fig, "pointwise")


gallery()
convergence()
examples()
pointwise()
