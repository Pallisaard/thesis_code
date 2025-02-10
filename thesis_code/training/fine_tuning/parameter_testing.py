import numpy as np
from opacus.accountants import RDPAccountant
import pandas as pd

parameter_info = {"delta": [], "sigma": [], "c": [], "n_steps": [], "epsilon": []}

sample_rate = 4 / 428
deltas = [1e-5]
sigmas = [1.0, 0.75, 1.5]
cs = [1.0, 0.75, 1.5]
alphas = None  #  = [1.1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]

setups = np.concatenate(
    (
        np.array([[1e-5 for i in range(2)], sigmas[0:2], [1.0 for i in range(2)]]).T,
        np.array([[1e-5 for i in range(2)], [1 for i in range(2)], cs[0:2]]).T,
    )
)
print("parameter setups:\n", setups)
print("number of setups:", len(setups))
print()

STEP_SIZE = 1

for delta, sigma, c in setups:
    n_steps = 0
    accountant = RDPAccountant()
    noise_multiplier = sigma * c

    epsilon = accountant.get_epsilon(sample_rate, delta)
    while epsilon < 2:
        for _ in range(STEP_SIZE):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        epsilon = accountant.get_epsilon(delta, alphas)
        n_steps += STEP_SIZE

    parameter_info["delta"].append(delta)
    parameter_info["sigma"].append(sigma)
    parameter_info["c"].append(c)
    parameter_info["n_steps"].append(n_steps)
    parameter_info["epsilon"].append(epsilon)

    print(
        f"Delta: {delta}, Sigma: {sigma}, C: {c}, Epsilon: {epsilon}, Steps: {n_steps}"
    )

    while epsilon < 5:
        for _ in range(STEP_SIZE):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        epsilon = accountant.get_epsilon(delta, alphas)
        n_steps += STEP_SIZE

    parameter_info["delta"].append(delta)
    parameter_info["sigma"].append(sigma)
    parameter_info["c"].append(c)
    parameter_info["n_steps"].append(n_steps)
    parameter_info["epsilon"].append(epsilon)
    print(
        f"Delta: {delta}, Sigma: {sigma}, C: {c}, Epsilon: {epsilon}, Steps: {n_steps}"
    )

    while epsilon < 10:
        for _ in range(STEP_SIZE):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        epsilon = accountant.get_epsilon(delta, alphas)
        n_steps += STEP_SIZE

    parameter_info["delta"].append(delta)
    parameter_info["sigma"].append(sigma)
    parameter_info["c"].append(c)
    parameter_info["n_steps"].append(n_steps)
    parameter_info["epsilon"].append(epsilon)

    print(
        f"Delta: {delta}, Sigma: {sigma}, C: {c}, Epsilon: {epsilon}, Steps: {n_steps}"
    )

print("\nFinal parameter info:")
print(parameter_info)

# Convert dictionary to pandas DataFrame and save as CSV
df = pd.DataFrame(parameter_info)
# df.to_csv("parameter_testing_results2.csv", index=False)
# print("\nResults saved to parameter_testing_results2.csv")
