"""
mppi_sampling.py
----------------
Interactive illustration of the MPPI sampling principle.

Setup
-----
  Input  : x ~ N(mu, sigma^2)
  Cost   : y = f(x) = x^2   (minimum at x = 0)
  Weights: w_k ∝ exp(−y_k / λ)   (cheaper samples get more weight)
  Update : mu_new = Σ w_k · x_k  (cost-weighted mean)

Key observations (drag the sliders to explore)
-----------------------------------------------
• Samples are coloured by MPPI weight: green = low cost, red = high cost.
• At the minimum (mu_x = 0):
    - The weight distribution is symmetric about x = 0.
    - mu_new = 0 = mu_x  ⟹  Δmu = 0  (fixed point / no update).
• Away from the minimum: the greener (cheaper) samples are on the side
  closer to zero, so the cost-weighted mean is pulled toward the minimum.
• Right panel — weighting function w(y) = exp(−y/λ):
    - Each sample dot lies exactly on this curve at its own cost value.
    - Small λ: steep curve → huge contrast between cheap and expensive samples.
    - Large λ: flat curve → all samples nearly equal weight, Δmu → 0.

Run
---
    python3 code/mppi_sampling.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider

# ── Parameters ────────────────────────────────────────────────────────────────

N      = 600
X_LIM  = (-4.5, 4.5)
Y_LIM  = (-0.5, 18.0)

# Fix the noise shape once; sliders shift/scale it so the cloud moves smoothly.
_z = np.random.default_rng(0).standard_normal(N)


def f(x):
    return x ** 2


# ── Figure layout ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(10, 8))
fig.patch.set_facecolor('white')
fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.12)

gs = gridspec.GridSpec(
    2, 2,
    width_ratios=[3.5, 1.2],
    height_ratios=[1.2, 3.5],
    hspace=0.05, wspace=0.08,
    figure=fig,
)
ax_main  = fig.add_subplot(gs[1, 0])
ax_top   = fig.add_subplot(gs[0, 0])
ax_right = fig.add_subplot(gs[1, 1])

# Three sliders
ax_sl_mu    = fig.add_axes([0.07, 0.04, 0.26, 0.025])
ax_sl_sigma = fig.add_axes([0.37, 0.04, 0.26, 0.025])
ax_sl_lam   = fig.add_axes([0.67, 0.04, 0.26, 0.025])
sl_mu    = Slider(ax_sl_mu,    r'$\mu_x$',    -3.5, 3.5, valinit=2.0, color='royalblue')
sl_sigma = Slider(ax_sl_sigma, r'$\sigma_x$',  0.1, 2.0, valinit=0.7, color='steelblue')
sl_lam   = Slider(ax_sl_lam,   r'$\lambda$',   0.1, 5.0, valinit=1.0, color='seagreen')

_xc = np.linspace(*X_LIM, 600)
_yc = f(_xc)


# ── Draw ──────────────────────────────────────────────────────────────────────

def redraw(mu, sigma, lam):
    for ax in (ax_main, ax_top, ax_right):
        ax.cla()

    # ── Compute ───────────────────────────────────────────────────────────────
    xs     = mu + sigma * _z
    ys     = f(xs)
    w_raw  = np.exp(-ys / lam)        # unnormalized MPPI weights
    w      = w_raw / w_raw.sum()      # normalized
    mu_new = float(w @ xs)            # cost-weighted mean (MPPI update)
    delta  = mu_new - mu
    f_mu   = float(f(np.float64(mu)))

    # Colour range based on unnormalized weights (same monotonic ordering)
    w_norm = (w_raw - w_raw.min()) / (w_raw.max() - w_raw.min() + 1e-14)

    # ── Main panel: parabola + weighted samples ───────────────────────────────
    ax_main.plot(_xc, _yc, 'k-', lw=2.5, zorder=2)
    ax_main.scatter(xs, ys, c=w_norm, cmap='RdYlGn',
                    s=14, alpha=0.55, vmin=0, vmax=1, zorder=5)

    ax_main.axhline(0, color='#cccccc', lw=0.7, zorder=1)
    ax_main.axvline(0, color='#cccccc', lw=0.7, zorder=1)

    # Current mean μ_x (blue)
    ax_main.axvline(mu,   color='royalblue', ls='--', lw=1.6, alpha=0.9)
    ax_main.axhline(f_mu, color='royalblue', ls='--', lw=1.6, alpha=0.9)
    ax_main.plot(mu, f_mu, 'o', color='royalblue', ms=11, zorder=10,
                 label=fr'$(\mu_x,\;f(\mu_x)) = ({mu:.2f},\;{f_mu:.2f})$')

    # MPPI updated mean μ_new (green)
    ax_main.axvline(mu_new, color='seagreen', ls='--', lw=1.6, alpha=0.9)
    ax_main.plot(mu_new, f(np.float64(mu_new)), '^',
                 color='seagreen', ms=11, zorder=10,
                 label=fr'$\mu_{{\rm new}} = {mu_new:.3f}$'
                       fr'  $(\Delta\mu = {delta:+.3f})$')

    ax_main.set_xlim(*X_LIM)
    ax_main.set_ylim(*Y_LIM)
    ax_main.set_xlabel('$x$', fontsize=13)
    ax_main.set_ylabel(r'$y = f(x) = x^2$', fontsize=13)
    ax_main.legend(loc='upper center', fontsize=10, framealpha=0.92)

    # ── Top panel: input Gaussian p(x) ───────────────────────────────────────
    xp = np.linspace(*X_LIM, 400)
    gx = np.exp(-0.5 * ((xp - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ax_top.fill_between(xp, gx, alpha=0.25, color='royalblue')
    ax_top.plot(xp, gx, color='royalblue', lw=2)
    ax_top.axvline(mu,     color='royalblue', ls='--', lw=1.6, alpha=0.9,
                   label=fr'$\mu_x = {mu:.2f}$')
    ax_top.axvline(mu_new, color='seagreen',  ls='--', lw=1.6, alpha=0.9,
                   label=fr'$\mu_{{\rm new}} = {mu_new:.2f}$')
    ax_top.set_xlim(*X_LIM)
    ax_top.set_yticks([])
    ax_top.set_ylabel('$p(x)$', fontsize=11)
    ax_top.tick_params(labelbottom=False)
    ax_top.legend(fontsize=9, loc='upper right', framealpha=0.92)

    # ── Right panel: MPPI weighting function w(y) = exp(−y/λ) ────────────────
    # Each sample dot lies exactly on this curve at its own cost value.
    y_w     = np.linspace(0, Y_LIM[1], 400)
    w_curve = np.exp(-y_w / lam)

    ax_right.fill_betweenx(y_w, w_curve, alpha=0.18, color='seagreen')
    ax_right.plot(w_curve, y_w, color='seagreen', lw=2.5,
                  label=r'$e^{-y/\lambda}$')

    # Sample dots at (exp(−y_k/λ), y_k) — lie exactly on the curve
    ax_right.scatter(np.exp(-ys / lam), ys, c=w_norm, cmap='RdYlGn',
                     s=14, alpha=0.55, vmin=0, vmax=1, zorder=5)

    # Horizontal line at f(μ_x) showing where the current mean maps
    ax_right.axhline(f_mu, color='royalblue', ls='--', lw=1.6, alpha=0.9)
    ax_right.plot(np.exp(-f_mu / lam), f_mu, 'o',
                  color='royalblue', ms=8, zorder=10,
                  label=fr'$f(\mu_x) = {f_mu:.2f}$')

    ax_right.set_xlim(-0.05, 1.15)
    ax_right.set_ylim(*Y_LIM)
    ax_right.set_xticks([0, 0.5, 1.0])
    ax_right.set_xlabel(r'$w = e^{-y/\lambda}$', fontsize=10)
    ax_right.tick_params(labelleft=False)
    ax_right.set_title('weighting fn', fontsize=9, pad=3)
    ax_right.legend(fontsize=8.5, loc='upper right', framealpha=0.92)

    fig.canvas.draw_idle()


sl_mu.on_changed(   lambda _: redraw(sl_mu.val, sl_sigma.val, sl_lam.val))
sl_sigma.on_changed(lambda _: redraw(sl_mu.val, sl_sigma.val, sl_lam.val))
sl_lam.on_changed(  lambda _: redraw(sl_mu.val, sl_sigma.val, sl_lam.val))

fig.suptitle(
    r'MPPI:  $x \sim \mathcal{N}(\mu_x,\,\sigma_x^2)$,  '
    r'$y = f(x) = x^2$,  '
    r'$w_k \propto e^{-y_k/\lambda}$,  '
    r'$\mu_{\rm new} = \sum w_k x_k$'
    '\n'
    r'green $=$ high weight (low cost) $\quad$ red $=$ low weight (high cost)',
    fontsize=10,
)

redraw(sl_mu.valinit, sl_sigma.valinit, sl_lam.valinit)
plt.show()
