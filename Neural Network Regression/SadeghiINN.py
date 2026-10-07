import torch
import torch.nn as nn
import torch.nn.functional as F


class SadeghiINN(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=8):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.prediction_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Linear(hidden_dim, 1)

        self.register_buffer("sigma_mean", torch.tensor(1.0))
        self.register_buffer("h", torch.tensor(1.0))
        self.register_buffer("calibrated", torch.tensor(False))

    def _raw(self, x):
        f = self.feature_extractor(x)
        y_hat = self.prediction_head(f)
        sigma = F.softplus(self.sigma_head(f))
        return y_hat, sigma

    def forward(self, x):
        y_hat, sigma = self._raw(x)
        if self.training or not bool(self.calibrated):
            # minibatch estimate of E[sigma]
            sigma_hat = sigma / sigma.mean()
        else:
            sigma_hat = sigma / self.sigma_mean          # fixed normaliser
        return y_hat, sigma_hat

    @staticmethod
    def heteroscedastic_loss(y_lower, y_upper, y_hat, sigma_hat):
        a = torch.abs(y_upper - y_hat) / sigma_hat
        b = torch.abs(y_lower - y_hat) / sigma_hat
        return torch.maximum(a.max(), b.max())

    def fit(self, loader, epochs=100, lr=1e-3, verbose=True):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            self.train()
            total = 0.0
            for xb, y_lo, y_hi in loader:
                y_hat, sigma_hat = self(xb)
                loss = self.heteroscedastic_loss(y_lo, y_hi, y_hat, sigma_hat)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"epoch {epoch + 1}: mean batch loss {total / len(loader):.4f}")
        return self

    @torch.no_grad()
    def calibrate(self, X, y_lower, y_upper):
        self.eval()
        y_hat, sigma = self._raw(X)
        self.sigma_mean.copy_(sigma.mean())
        self.calibrated.fill_(True)
        sigma_hat = sigma / self.sigma_mean
        self.h.copy_(self.heteroscedastic_loss(
            y_lower, y_upper, y_hat, sigma_hat))
        return self

    @torch.no_grad()
    def predict_interval(self, x):
        assert bool(self.calibrated), "call calibrate() first"
        self.eval()
        y_hat, sigma = self._raw(x)
        half_width = self.h * sigma / self.sigma_mean    # h * sigma_hat(x)
        return y_hat - half_width, y_hat + half_width
