import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

np.random.seed(0)
data = np.concatenate([np.random.normal(2, 1.0, 300),
                       np.random.normal(8, 1.5, 200)])

mu = np.array([1.0, 5.0])
sigma = np.array([1.0, 1.0])
pi = np.array([0.5, 0.5])

for it in range(1, 101):
    resp = np.array([pi[k] * norm.pdf(data, mu[k], sigma[k]) for k in range(2)])
    resp /= resp.sum(axis=0)
    Nk = resp.sum(axis=1)
    mu_new = (resp * data).sum(axis=1) / Nk
    sigma = np.sqrt((resp * (data - mu_new[:, None]) ** 2).sum(axis=1) / Nk)
    pi = Nk / len(data)
    if np.abs(mu_new - mu).max() < 1e-6:
        mu = mu_new
        break
    mu = mu_new

print(f"Converged after {it} iterations")
print("Means   :", mu.round(3))
print("Std devs:", sigma.round(3))
print("Weights :", pi.round(3))

gm = GaussianMixture(2, random_state=0).fit(data.reshape(-1, 1))
print("\nsklearn GMM means:", np.sort(gm.means_.ravel()).round(3))
