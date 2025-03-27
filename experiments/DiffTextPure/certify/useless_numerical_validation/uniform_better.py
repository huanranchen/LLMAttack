for V in [2, 3, 4, 5, 6, 10, 100, 1000, 10000]:
    beta_bar = 0.5
    beta = (1 - beta_bar) / (V - 1)
    left = (
        beta_bar**3
        + 3 * beta_bar**2 * beta
        + 3 * (V - 2) * beta_bar**2 * beta
        + 3 * (V - 2) ** 2 * beta_bar * beta**2
    )
    right = 1 - (1 - beta_bar) ** 3
    print(left, right)
