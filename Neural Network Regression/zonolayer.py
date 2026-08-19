import numpy as np
import PyIPM


# def compute_zonolayer_bounds(latent_train, y_lower, y_upper, latent_test):
#     X = np.atleast_2d(np.asarray(latent_train, dtype=np.float64))
#     X_test = np.atleast_2d(np.asarray(latent_test, dtype=np.float64))

#     y_l = np.asarray(y_lower, dtype=np.float64).ravel()
#     y_u = np.asarray(y_upper, dtype=np.float64).ravel()

#     X = np.hstack([X, np.ones((X.shape[0], 1))])
#     # print(X.shape)
#     X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

#     c_y = 0.5 * (y_u + y_l)
#     r_y = 0.5 * (y_u - y_l)

#     # Beta = np.linalg.pinv(X)
#     XtX = X.T @ X
#     Beta = np.linalg.solve(XtX, X.T)

#     H = X_test @ Beta
#     w_c = Beta @ c_y

#     c_test = X_test @ w_c
#     G_test = H * r_y

#     radius = np.sum(np.abs(G_test), axis=1)

#     return {
#         "midpoint": c_test,
#         "generator": G_test,
#         "radius": radius,
#         "y_lower": c_test - radius,
#         "y_upper": c_test + radius,
#     }

def fit_zonolayer(latent_train, y_lower, y_upper):
    X = np.atleast_2d(np.asarray(latent_train, dtype=np.float64))
    X = np.hstack([X, np.ones((X.shape[0], 1))])

    y_l = np.asarray(y_lower, dtype=np.float64).ravel()
    y_u = np.asarray(y_upper, dtype=np.float64).ravel()

    c_y = 0.5 * (y_u + y_l)
    r_y = 0.5 * (y_u - y_l)

    XtX = X.T @ X
    Beta = np.linalg.solve(XtX, X.T)   # (d+1, n_train)
    w_c = Beta @ c_y                    # (d+1,)

    return {"Beta": Beta, "r_y": r_y, "w_c": w_c}


def predict_zonolayer(fitted, latent_test):
    X_test = np.atleast_2d(np.asarray(latent_test, dtype=np.float64))
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

    Beta, r_y, w_c = fitted["Beta"], fitted["r_y"], fitted["w_c"]

    H = X_test @ Beta
    c_test = X_test @ w_c
    G_test = H * r_y
    radius = np.sum(np.abs(G_test), axis=1)

    return {
        "midpoint": c_test,
        "generator": G_test,
        "radius": radius,
        "y_lower": c_test - radius,
        "y_upper": c_test + radius,
    }


def compute_ipm_bounds(x_train, x_test, y_lower_train, y_upper_train):
    ipm_model = PyIPM.IPM()

    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    y_lower = np.asarray(y_lower_train, dtype=np.float64).flatten()
    y_upper = np.asarray(y_upper_train, dtype=np.float64).flatten()

    # Combination of Endpoints
    x_train = np.vstack([x_train, x_train])
    y_train = np.concatenate([y_lower, y_upper])

    ipm_model.fit(x_train, y_train)

    return ipm_model.predict(x_test)
