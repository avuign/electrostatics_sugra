import matplotlib
import numpy as np
from scipy.integrate import quad

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solver import charge_density

plt.rcParams.update({"font.family": "serif", "font.size": 12})


def convergence_with_Nmax():
    Rs, zs = 10.0, 1.0
    r_plot = np.linspace(0, Rs, 300)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for Nmax in [2, 4, 6, 8, 10]:
        print(f"Fig 7: Nmax = {Nmax}")
        sigma = charge_density(Rs, zs, Nmax)
        sigma.minimize()
        y = np.array([sigma(r) for r in r_plot])
        axes[0].plot(r_plot, y, label=rf"$N_{{\max}}={Nmax}$")
        mask = r_plot > 9.0
        axes[1].plot(r_plot[mask], y[mask], label=rf"$N_{{\max}}={Nmax}$")

    for ax in axes:
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$\sigma(r)$")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[0].set_title(rf"Charge density ($z_s={zs}$, $R_s={Rs}$)")
    axes[1].set_title("Zoom near tip")
    plt.tight_layout()
    plt.savefig("fig7.png", dpi=150)
    print("Saved fig7.png")


def sigma():
    Rs = 1.0
    Nmax = 4
    r_norm = np.linspace(1e-4, 1 - 1e-4, 300)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for zs in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        print(f"Fig 8: zs = {zs}")
        sigma = charge_density(Rs, zs, Nmax)
        sigma.minimize()
        y = np.array([sigma(r * Rs) for r in r_norm])
        y0 = sigma(1e-6)
        if abs(y0) > 1e-15:
            ax.plot(r_norm, y / y0, label=rf"$z_s = {zs}$")

    r_an = np.linspace(0, 1, 200)
    ax.plot(r_an, 1 - r_an**2, "k--", lw=1.5, label=r"$1-r^2$")
    ax.plot(
        r_an,
        np.sqrt(np.maximum(1 - r_an**2, 0)),
        "r--",
        lw=1.5,
        label=r"$\sqrt{1-r^2}$",
    )

    ax.set_xlabel(r"$r/R_s$")
    ax.set_ylabel(r"$\sigma(r)/\sigma(0)$")
    ax.legend(fontsize=9, ncol=2)
    ax.set_title("Normalized charge density (Figure 8)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig8.png", dpi=150)
    print("Saved fig8.png")


def charge_and_potential():
    Rs, Vb, Nmax = 1.0, 1.0, 4
    ratios = np.logspace(-2, 1.5, 25)

    Qs_list, Vt_list, zs_list = [], [], []

    for ratio in ratios:
        zs = ratio * Rs
        print(f"Fig 9: zs/Rs = {ratio:.4f}")
        sigma = charge_density(Rs, zs, Nmax)
        sigma.minimize()
        Qs_list.append(sigma.total_charge())
        Vt_list.append(sigma.Vs + Vb * zs**3)
        zs_list.append(zs)

    rat = np.array(ratios)
    Qs = np.array(Qs_list)
    Vt = np.array(Vt_list)
    zs_arr = np.array(zs_list)

    rat_an = np.logspace(-2.5, 2, 300)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Qs
    norm_Q = 15 / (8 * np.pi * Vb * Rs**5)
    axes[0].loglog(rat, norm_Q * Qs, "bo", ms=5, label="Numerics")
    axes[0].axhline(
        1, color="k", ls="--", lw=1, label=r"$Q_s = \frac{8\pi}{15} V_b R_s^5$"
    )
    Qs_large = 3 * np.pi**2 / 2 * Vb * Rs**2 * (rat_an * Rs)
    axes[0].loglog(
        rat_an,
        norm_Q * Qs_large,
        "r--",
        lw=1,
        label=r"$Q_s = \frac{3\pi^2}{2} V_b R_s^2 z_s$",
    )
    axes[0].set_xlabel(r"$z_s/R_s$")
    axes[0].set_ylabel(r"$\frac{15}{8\pi V_b R_s^5}\, Q_s$")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].set_title(r"Total charge $Q_s$")

    # Right: Vtilde
    axes[1].semilogx(rat, Vt / (Vb * zs_arr * Rs**2), "bo", ms=5, label="Numerics")
    axes[1].axhline(1, color="k", ls="--", lw=1, label=r"$\tilde{V}_s = V_b z_s R_s^2$")
    axes[1].axhline(
        1.5,
        color="r",
        ls="--",
        lw=1,
        label=r"$\tilde{V}_s = \frac{3}{2} V_b z_s R_s^2$",
    )
    axes[1].set_xlabel(r"$z_s/R_s$")
    axes[1].set_ylabel(r"$\frac{1}{V_b z_s R_s^2}\, \tilde{V}_s$")
    axes[1].set_ylim(0.9, 1.6)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].set_title(r"Potential $\tilde{V}_s$")

    plt.suptitle("Figure 9", fontsize=14)
    plt.tight_layout()
    plt.savefig("fig9.png", dpi=150)
    print("Saved fig9.png")


def octopole():
    Rs, Vb, Nmax = 1.0, 1.0, 4
    ratios = np.logspace(-2, 1.5, 25)

    q3_list, Qs_list, zs_list = [], [], []

    for ratio in ratios:
        zs = ratio * Rs
        print(f"Fig 10: zs/Rs = {ratio:.4f}")
        sigma = charge_density(Rs, zs, Nmax)
        sigma.minimize()
        Qs = sigma.total_charge()
        # octopole: q3 = 2 zs^3 Qs - 6 zs * int r^2 f(r) dr
        # with int r^2 f(r) dr = (4pi/3) int_0^Rs r^4 sigma(r) dr
        I4 = quad(lambda r: r**4 * sigma(r), 0, Rs)[0]
        q3 = 2 * zs**3 * Qs - 6 * zs * (4 * np.pi / 3) * I4
        q3_list.append(q3)
        Qs_list.append(Qs)
        zs_list.append(zs)

    rat = np.array(ratios)
    q3 = np.array(q3_list)
    zs_arr = np.array(zs_list)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    norm = 35 / (16 * np.pi * Vb * Rs**7)
    ax.loglog(rat, np.abs(norm * q3 / zs_arr), "bo", ms=5, label="Numerics")
    ax.axhline(
        1, color="k", ls="--", lw=1, label=r"$q_3 = -\frac{16\pi}{35} V_b R_s^7 z_s$"
    )
    rat_an = np.logspace(-2.5, 2, 300)
    Qs_large = 3 * np.pi**2 / 2 * Vb * Rs**2 * (rat_an * Rs)
    q3_large = 2 * Qs_large * (rat_an * Rs) ** 3
    ax.loglog(
        rat_an,
        np.abs(norm * q3_large / (rat_an * Rs)),
        "r--",
        lw=1,
        label=r"$q_3 = 2 Q_s z_s^3$",
    )
    ax.set_xlabel(r"$z_s/R_s$")
    ax.set_ylabel(r"$\frac{35}{16\pi V_b R_s^7 z_s}\, |q_{3,s}|$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title("Octopole $q_{3,s}$ (Figure 10)")
    plt.tight_layout()
    plt.savefig("fig10.png", dpi=150)
    print("Saved fig10.png")


if __name__ == "__main__":
    convergence_with_Nmax()
    sigma()
    charge_and_potential()
    octopole()
