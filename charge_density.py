import numpy as np
from scipy.integrate import quad
from scipy.special import eval_legendre

NGRID = 10


class charge_density:
    def __init__(self, Rs, zs, Nmax):
        self.Rs = Rs
        self.zs = zs
        self.Nmax = Nmax
        self.params = np.zeros(Nmax + 2)
        self.Vs = 0
        self.V_matrix = self._compute_potential_matrix()

    # compute V_k(r_i)
    def _compute_potential_matrix(self):
        n_basis = len(self.params)
        grid = [i * self.Rs / NGRID for i in range(1, NGRID + 1)]

        def basis(k, u):
            if k < n_basis - 1:
                return eval_legendre(k, 2 * u / self.Rs - 1)
            return np.sqrt(self.Rs**2 - u**2)

        def integrand(u, k, ri):
            diff2 = (ri - u) ** 2
            return (
                u
                * basis(k, u)
                * np.log(
                    (1 + 4 * ri * u / diff2)
                    / (1 + 4 * ri * u / (diff2 + 4 * self.zs**2))
                )
            )

        return np.array(
            [
                [
                    quad(integrand, 0, self.Rs, args=(k, ri), points=[ri])[0]
                    / (4 * np.pi * ri)
                    for ri in grid
                ]
                for k in range(n_basis)
            ]
        )

    def __call__(self, r):
        poly = []
        for i in range(0, len(self.params) - 1):
            poly.append(self.params[i] * eval_legendre(i, 2 * r / self.Rs - 1))
        return sum(poly) + self.params[-1] * np.sqrt(self.Rs**2 - r**2)

    def get_constraint(self):
        constraint = []

        for i in range(1, NGRID + 1):
            ri = i * self.Rs / NGRID
            ckVk = []
            for k in range(0, len(self.params)):
                ckVk.append(self.params[k] * self.V_matrix[k, i - 1])

            constraint.append(
                (sum(ckVk) - (self.zs**3 - self.zs * ri**2) - self.Vs) ** 2
            )
        derivative_at_zero = []
        for i in range(0, len(self.params) - 1):
            derivative_at_zero.append(
                self.params[i] * (-1) ** (i - 1) * i * (i + 1) / 2 * 2 / self.Rs
            )

        constraint.append(self(self.Rs) ** 2)
        constraint.append(sum(derivative_at_zero) ** 2)

        return sum(constraint)

    def total_charge(self):
        return quad(lambda r: 4 * np.pi * r**2 * self(r), 0, self.Rs)[0]

    def minimize(self):
        n_basis = len(self.params)
        n_unknowns = n_basis + 1  # params + Vs
        n_eqs = NGRID + 2

        A = np.zeros((n_eqs, n_unknowns))
        b = np.zeros(n_eqs)

        # EOM rows
        for i in range(NGRID):
            ri = (i + 1) * self.Rs / NGRID
            A[i, :n_basis] = self.V_matrix[:, i]
            A[i, -1] = -1.0
            b[i] = self.zs**3 - self.zs * ri**2

        # sigma(Rs) = 0
        for k in range(n_basis - 1):
            A[NGRID, k] = 1.0  # P_n(1) = 1
        A[NGRID, n_basis - 1] = 0.0  # sqrt(Rs^2 - Rs^2) = 0

        # sigma'(0) = 0
        for k in range(n_basis - 1):
            A[NGRID + 1, k] = (2.0 / self.Rs) * (-1) ** (k - 1) * k * (k + 1) / 2.0

        # solve
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        self.params = x[:n_basis]
        self.Vs = x[-1]
