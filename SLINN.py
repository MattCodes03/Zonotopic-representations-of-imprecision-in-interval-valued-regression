import numpy as np
from dataclasses import dataclass


@dataclass
class Interval:
    mid: np.ndarray
    rad: np.ndarray = None

    def __post_init__(self):
        self.mid = np.asarray(self.mid, dtype=float)
        if self.rad is None:
            self.rad = np.zeros_like(self.mid)
        else:
            self.rad = np.abs(np.asarray(self.rad, dtype=float))
            if self.rad.shape != self.mid.shape:
                # Standard interval broadcast: use the max or mean radius
                self.rad = np.full_like(self.mid, np.mean(self.rad))

    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(self.mid + other.mid, self.rad + other.rad)
        return Interval(self.mid + other, self.rad)

    def __sub__(self, other):
        if isinstance(other, Interval):
            return Interval(self.mid - other.mid, self.rad + other.rad)
        return Interval(self.mid - other, self.rad)

    def __matmul__(self, other):
        # Implementation of Mid-Rad matrix multiplication
        if not isinstance(other, Interval):
            other = Interval(other, np.zeros_like(other))
        nm = self.mid @ other.mid
        # Formula: rad(AB) = |mid(A)|rad(B) + rad(A)|mid(B)| + rad(A)rad(B)
        nr = np.abs(self.mid) @ other.rad + \
            self.rad @ np.abs(other.mid) + \
            self.rad @ other.rad
        return Interval(nm, nr)


def midrad(mid, rad=0.0):
    """
    Helper function to create an Interval object,
    matching the MATLAB Intlab syntax.
    """
    mid_array = np.asarray(mid, dtype=float)
    # Create a radius array of the same shape filled with the rad value
    rad_array = np.full_like(mid_array, rad)
    return Interval(mid_array, rad_array)


class SLINN:
    def __init__(self, nn_input_dim, nn_hdim, nn_output_dim, L=2, L_rate=0.01, Beta=0.9):
        self.L, self.L_rate, self.Beta = L, L_rate, Beta
        self.nn_hdim = nn_hdim
        self.nn_output_dim = nn_output_dim
        # Using 1-based indexing internally to match your logic/MATLAB
        self.weights = [None] * (L + 2)
        self.biases = [None] * (L + 2)
        self.MdLw = [0.0] * (L + 2)
        self.MdLb = [0.0] * (L + 2)

        for i in range(2, L + 2):
            d_in = nn_input_dim if i == 2 else nn_hdim
            d_out = nn_output_dim if i == L + 1 else nn_hdim
            sig = np.sqrt(2 / (d_in + d_out))
            self.weights[i] = np.random.normal(0, sig, (d_in, d_out))
            self.biases[i] = np.zeros((1, d_out))
            self.MdLw[i] = np.zeros_like(self.weights[i])
            self.MdLb[i] = np.zeros_like(self.biases[i])

        self.dhdYi_final = None

    def train(self, x, Y, epochs=9000):
        n_samples = x.shape[0]
        I = np.eye(self.nn_hdim)
        ones_vec = np.ones((n_samples, 1))

        # Consistent buffers
        dWdYi_past = [np.zeros((self.nn_hdim, n_samples)), np.zeros(
            (self.nn_hdim, n_samples))]
        dbdYi_past = [np.zeros((1, n_samples)), np.zeros((1, n_samples))]
        dSdY_old = np.zeros((self.nn_hdim, n_samples))
        dDdY_old = np.zeros((1, n_samples))
        WTail_old = 0.0
        bTail_old = 0.0

        # MATLAB initialization: weights{L+1} and biases{L+1} start as intervals
        self.weights[self.L+1] = Interval(self.weights[self.L+1])
        self.biases[self.L+1] = Interval(self.biases[self.L+1])

        h = [None] * (self.L + 2)
        z = [None] * (self.L + 2)

        for k in range(1, epochs + 1):
            # --- Forward Pass (Midpoint) ---
            h[1] = x
            for i in range(2, self.L + 1):
                z[i] = h[i-1] @ self.weights[i] + self.biases[i]
                h[i] = np.tanh(z[i])
            h[self.L+1] = h[self.L] @ self.weights[self.L+1].mid + \
                self.biases[self.L+1].mid

            # --- Backprop (Momentum) ---
            G = (h[self.L+1] - Y.mid)
            dLw_fin = h[self.L].T @ G
            dLb_fin = np.sum(G, axis=0, keepdims=True)

            self.MdLw[self.L+1] = (1-self.Beta) * \
                dLw_fin + self.Beta*self.MdLw[self.L+1]
            self.MdLb[self.L+1] = (1-self.Beta) * \
                dLb_fin + self.Beta*self.MdLb[self.L+1]

            G_back = G @ self.weights[self.L+1].mid.T
            for i in range(self.L, 1, -1):
                D = G_back * (1 - np.tanh(z[i])**2)
                self.MdLw[i] = (1-self.Beta)*(h[i-1].T @
                                              D) + self.Beta*self.MdLw[i]
                self.MdLb[i] = (1-self.Beta)*np.sum(D, axis=0,
                                                    keepdims=True) + self.Beta*self.MdLb[i]
                G_back = D @ self.weights[i].T

            # --- Updates ---
            for j in range(2, self.L + 2):
                if j == self.L + 1:
                    self.weights[j].mid -= self.L_rate * self.MdLw[j]
                    self.biases[j].mid -= self.L_rate * self.MdLb[j]
                else:
                    self.weights[j] -= self.L_rate * self.MdLw[j]
                    self.biases[j] -= self.L_rate * self.MdLb[j]

            # --- Sensitivity Recurrence ---
            dS_curr = -h[self.L].T
            dD_curr = -ones_vec.T
            LRB = self.L_rate * (1 - self.Beta)

            dWdY = -LRB * (dS_curr + self.Beta * dSdY_old)
            dbdY = -LRB * (dD_curr + self.Beta * dDdY_old)

            DW1_0 = I - LRB * (h[self.L].T @ h[self.L])
            DW1_1 = -LRB * (h[self.L].T @ ones_vec)
            Db1_0 = -LRB * np.sum(h[self.L], axis=0, keepdims=True)
            Db1_1 = np.array([[1 - LRB * n_samples]])

            DW2_0 = -LRB * self.Beta * (h[self.L].T @ h[self.L])
            DW2_1 = -LRB * self.Beta * (h[self.L].T @ ones_vec)
            Db2_0 = -LRB * self.Beta * \
                np.sum(h[self.L], axis=0, keepdims=True)
            Db2_1 = np.array([[-LRB * self.Beta * n_samples]])

            WTail_new = DW2_0 @ dWdYi_past[1] + \
                DW2_1 @ dbdYi_past[1] + self.Beta * WTail_old
            bTail_new = Db2_0 @ dWdYi_past[1] + \
                Db2_1 @ dbdYi_past[1] + self.Beta * bTail_old

            dWdYi_n = dWdY + \
                DW1_0 @ dWdYi_past[0] + DW1_1 @ dbdYi_past[0] + WTail_new
            dbdYi_n = dbdY + \
                Db1_0 @ dWdYi_past[0] + Db1_1 @ dbdYi_past[0] + bTail_new

            # --- Interval Correction ---
            # IMPORTANT: The weight radius is calculated by multiplying sensitivity by the input radius.
            # To match MATLAB's `corrW = dWdYi_n * (Y - Y.mid)`, we use absolute multiplication.
            self.weights[self.L+1].rad = np.abs(dWdYi_n) @ Y.rad
            self.biases[self.L+1].rad = np.abs(dbdYi_n) @ Y.rad

            # Shift buffers
            dWdYi_past[1], dWdYi_past[0] = dWdYi_past[0], dWdYi_n
            dbdYi_past[1], dbdYi_past[0] = dbdYi_past[0], dbdYi_n
            dSdY_old = dS_curr + self.Beta * dSdY_old
            dDdY_old = dD_curr + self.Beta * dDdY_old
            WTail_old, bTail_old = WTail_new, bTail_new

            # Save the final sensitivity components for prediction
            self.dW_sens = dWdYi_n
            self.db_sens = dbdYi_n

    def predict(self, x, Y_train_obj):
        # Forward pass to get h_L for the new x (x_grid)
        h = x
        for i in range(2, self.L + 1):
            h = np.tanh(h @ self.weights[i] + self.biases[i])

            # Midpoint prediction
        res_mid = h @ self.weights[self.L+1].mid + self.biases[self.L+1].mid

        # Interval prediction:
        # In MATLAB, h_int = h_n{L+1} + dhdYi_n * (Y - Y.mid)
        # dhdYi_n = h_L * dWdYi_past + dbdYi_past
        # Therefore, rad = |h_L @ dW_sens + db_sens| @ Y_train_rad

        dhdYi = h @ self.dW_sens + self.db_sens
        res_rad = np.abs(dhdYi) @ Y_train_obj.rad

        return Interval(res_mid, res_rad)
