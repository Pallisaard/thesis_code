import numpy as np
from opacus.accountants import RDPAccountant
import pandas as pd
from multiprocessing import Pool
import os


def process_parameters(setup):
    delta, sigma, c = setup
    results = []
    n_steps = 0
    accountant = RDPAccountant()
    noise_multiplier = sigma * c
    sample_rate = 4 / 428  # Moved inside function since it's needed here
    # alphas = [1.1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]  # Moved inside function
    alphas = [1.1, 2, 3, 5, 10, 20, 50, 100]
    STEP_SIZE = 1

    epsilon = accountant.get_epsilon(sample_rate, delta)

    # Track progress to epsilon = 2
    while epsilon < 2:
        for _ in range(STEP_SIZE):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        epsilon = accountant.get_epsilon(delta, alphas)
        n_steps += STEP_SIZE

    results.append((delta, sigma, c, n_steps, epsilon))
    print(
        f"Process {os.getpid()}: Delta: {delta}, Sigma: {sigma}, C: {c}, Epsilon: {epsilon}, Steps: {n_steps}"
    )

    # Track progress to epsilon = 5
    while epsilon < 5:
        for _ in range(STEP_SIZE):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        epsilon = accountant.get_epsilon(delta, alphas)
        n_steps += STEP_SIZE

    results.append((delta, sigma, c, n_steps, epsilon))
    print(
        f"Process {os.getpid()}: Delta: {delta}, Sigma: {sigma}, C: {c}, Epsilon: {epsilon}, Steps: {n_steps}"
    )

    # Track progress to epsilon = 10
    while epsilon < 10:
        for _ in range(STEP_SIZE):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        epsilon = accountant.get_epsilon(delta, alphas)
        n_steps += STEP_SIZE

    results.append((delta, sigma, c, n_steps, epsilon))
    print(
        f"Process {os.getpid()}: Delta: {delta}, Sigma: {sigma}, C: {c}, Epsilon: {epsilon}, Steps: {n_steps}"
    )

    return results


def main():
    deltas = [1e-3, 1e-5, 1e-7]
    sigmas = [0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    cs = sigmas.copy()

    # Create parameter setups
    delta_setups = np.array([[delta, 1.0, 1.0] for delta in deltas])
    c_setups = np.array([[1e-5, 1.0, c] for c in cs])
    sigma_setups = np.array([[1e-5, sigma, 1.0] for sigma in sigmas])
    setups = np.concatenate((delta_setups, c_setups, sigma_setups))

    print("parameter setups:\n", setups)
    print("number of setups:", len(setups))
    print()

    # Initialize the parameter info dictionary
    parameter_info = {"delta": [], "sigma": [], "c": [], "n_steps": [], "epsilon": []}

    # Use multiprocessing to process parameter setups
    num_processes = os.cpu_count()  # Use number of CPU cores
    with Pool(processes=num_processes) as pool:
        all_results = pool.map(process_parameters, setups)

    # Flatten results and update parameter_info
    for setup_results in all_results:
        for delta, sigma, c, n_steps, epsilon in setup_results:
            parameter_info["delta"].append(delta)
            parameter_info["sigma"].append(sigma)
            parameter_info["c"].append(c)
            parameter_info["n_steps"].append(n_steps)
            parameter_info["epsilon"].append(epsilon)

    print("\nFinal parameter info:")
    print(parameter_info)

    # Convert dictionary to pandas DataFrame and save as CSV
    df = pd.DataFrame(parameter_info)
    df.to_csv("parameter_testing_results_out_alphas.csv", index=False)
    print("\nResults saved to parameter_testing_results_out_alphas.csv")


if __name__ == "__main__":
    main()
