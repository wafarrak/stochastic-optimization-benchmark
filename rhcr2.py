import numpy as np
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

LOWER, UPPER = -512, 512



# Benchmark Functions


def frog(x):
    x1, x2 = x
    term1 = x1 * np.sin(np.sqrt(abs(x2 + 1 - x1))) * np.cos(np.sqrt(abs(x1 + x2 + 1)))
    term2 = (x2 + 1) * np.cos(np.sqrt(abs(x2 + 1 - x1))) * np.sin(np.sqrt(abs(x1 + x2 + 1)))
    return term1 + term2


def rastrigin(x):
    x1, x2 = x
    return 20 + x1**2 + x2**2 - 10 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))


def ackley(x):
    x1, x2 = x
    return (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2)))
        + np.e + 20
    )


FUNCTIONS = {
    "Frog": {
        "func": frog,
        "bounds": (-512, 512),
        "start": (-200, 200),
        "z": 120,
        "p": 600,
        "target": -512.0,
        "success_threshold": -510
    },
    "Rastrigin": {
        "func": rastrigin,
        "bounds": (-5.12, 5.12),
        "start": (4, 4),
        "z": 1.5,
        "p": 300,
        "target": 0.0,
        "success_threshold": 1.0
    },
    "Ackley": {
        "func": ackley,
        "bounds": (-32.768, 32.768),
        "start": (20, 20),
        "z": 8,
        "p": 300,
        "target": 0.0,
        "success_threshold": 1.0
    }
}



# Helpers



def clip_point(x, lower, upper):
    return np.clip(x, lower, upper)


def percent_error(value, target):
    if target == 0:
        return abs(value - target)
    return abs(value - target) / abs(target) * 100




# Basic Randomized Hill Climbing



def RHC(func, start, z, p, seed, bounds, max_iter=200, budget=None, track=False):
    random.seed(seed)
    np.random.seed(seed)

    lower, upper = bounds

    current = np.array(start, dtype=float)
    current_val = func(current)
    calls = 1

    history = [current_val]
    path = [current.copy()]

    start_time = time.time()

    for _ in range(max_iter):
        if budget is not None and calls >= budget:
            break

        best_neighbor = current.copy()
        best_val = current_val

        for _ in range(p):
            if budget is not None and calls >= budget:
                break

            neighbor = current + np.array([
                random.uniform(-z, z),
                random.uniform(-z, z)
            ])

            neighbor = clip_point(neighbor, lower, upper)
            val = func(neighbor)
            calls += 1

            if val < best_val:
                best_neighbor = neighbor.copy()
                best_val = val

        if best_val < current_val:
            current = best_neighbor
            current_val = best_val
            history.append(current_val)
            path.append(current.copy())
        else:
            break

    runtime = time.time() - start_time

    result = {
        "solution": current,
        "value": current_val,
        "calls": calls,
        "runtime": runtime
    }

    if track:
        result["history"] = history
        result["path"] = np.array(path)

    return result



# Improved 5-Stage RHCR


def RHCR5(func, start, z, p, seed, bounds, budget=None, track=False):
    shrink_factors = [1, 10, 50, 250, 1000]

    current_start = start
    total_calls = 0
    total_runtime = 0

    all_history = []
    all_path = []

    stage_solutions = []
    stage_values = []
    stage_calls = []

    for i, factor in enumerate(shrink_factors):
        remaining_budget = None
        if budget is not None:
            remaining_budget = budget - total_calls
            if remaining_budget <= 0:
                break

        result = RHC(
            func=func,
            start=current_start,
            z=z / factor,
            p=p,
            seed=seed + i,
            bounds=bounds,
            budget=remaining_budget,
            track=track
        )

        current_start = result["solution"]
        total_calls += result["calls"]
        total_runtime += result["runtime"]

        stage_solutions.append(result["solution"])
        stage_values.append(result["value"])
        stage_calls.append(result["calls"])

        if track:
            all_history.extend(result["history"])
            if len(all_path) == 0:
                all_path = result["path"]
            else:
                all_path = np.vstack([all_path, result["path"]])

    output = {
        "solution": stage_solutions[-1],
        "value": stage_values[-1],
        "calls": total_calls,
        "runtime": total_runtime,
        "stage_solutions": stage_solutions,
        "stage_values": stage_values,
        "stage_calls": stage_calls
    }

    if track:
        output["history"] = all_history
        output["path"] = np.array(all_path)

    return output



# Random Search


def random_search(func, seed, bounds, budget=7203, track=False):
    random.seed(seed)
    np.random.seed(seed)

    lower, upper = bounds

    best = np.array([
        random.uniform(lower, upper),
        random.uniform(lower, upper)
    ])

    best_val = func(best)
    history = [best_val]
    path = [best.copy()]

    start_time = time.time()

    for _ in range(budget - 1):
        candidate = np.array([
            random.uniform(lower, upper),
            random.uniform(lower, upper)
        ])

        val = func(candidate)

        if val < best_val:
            best = candidate.copy()
            best_val = val
            path.append(best.copy())

        history.append(best_val)

    runtime = time.time() - start_time

    result = {
        "solution": best,
        "value": best_val,
        "calls": budget,
        "runtime": runtime
    }

    if track:
        result["history"] = history
        result["path"] = np.array(path)

    return result



# Simulated Annealing


def simulated_annealing(func, start, z, seed, bounds, budget=7203, temp=100, cooling=0.995, track=False):
    random.seed(seed)
    np.random.seed(seed)

    lower, upper = bounds

    current = np.array(start, dtype=float)
    current_val = func(current)

    best = current.copy()
    best_val = current_val

    calls = 1
    history = [best_val]
    path = [best.copy()]

    start_time = time.time()

    for _ in range(budget - 1):
        neighbor = current + np.array([
            random.uniform(-z, z),
            random.uniform(-z, z)
        ])

        neighbor = clip_point(neighbor, lower, upper)
        neighbor_val = func(neighbor)
        calls += 1

        delta = neighbor_val - current_val

        if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-12)):
            current = neighbor.copy()
            current_val = neighbor_val

        if current_val < best_val:
            best = current.copy()
            best_val = current_val
            path.append(best.copy())

        temp *= cooling
        history.append(best_val)

    runtime = time.time() - start_time

    result = {
        "solution": best,
        "value": best_val,
        "calls": calls,
        "runtime": runtime
    }

    if track:
        result["history"] = history
        result["path"] = np.array(path)

    return result



# Benchmark Experiment


def benchmark_all(num_seeds=50, budget=7203):
    rows = []

    for function_name, config in FUNCTIONS.items():
        func = config["func"]
        bounds = config["bounds"]
        start = config["start"]
        z = config["z"]
        p = config["p"]
        target = config["target"]
        threshold = config["success_threshold"]

        for seed in range(num_seeds):

            rhc = RHC(func, start, z, p, seed, bounds, budget=budget)
            rhcr5 = RHCR5(func, start, z, p, seed, bounds, budget=budget)
            rs = random_search(func, seed, bounds, budget=budget)
            sa = simulated_annealing(func, start, z, seed, bounds, budget=budget)

            results = [
                ("Basic RHC", rhc),
                ("RHCR5", rhcr5),
                ("Random Search", rs),
                ("Simulated Annealing", sa)
            ]

            for algorithm, result in results:
                rows.append({
                    "function": function_name,
                    "algorithm": algorithm,
                    "seed": seed,
                    "final_value": result["value"],
                    "calls": result["calls"],
                    "runtime_seconds": result["runtime"],
                    "x": result["solution"][0],
                    "y": result["solution"][1],
                    "target": target,
                    "error": abs(result["value"] - target),
                    "percent_error": percent_error(result["value"], target),
                    "success": result["value"] <= threshold
                })

    df = pd.DataFrame(rows)
    df.to_csv("full_algorithm_benchmark_results.csv", index=False)
    return df



# Summary Tables


def create_summary(df):
    summary = df.groupby(["function", "algorithm"]).agg(
        best_value=("final_value", "min"),
        mean_value=("final_value", "mean"),
        std_value=("final_value", "std"),
        median_value=("final_value", "median"),
        mean_calls=("calls", "mean"),
        mean_runtime_seconds=("runtime_seconds", "mean"),
        success_rate=("success", "mean"),
        best_error=("error", "min"),
        mean_error=("error", "mean"),
        best_percent_error=("percent_error", "min")
    ).reset_index()

    summary["success_rate"] = summary["success_rate"] * 100

    summary.to_csv("full_algorithm_benchmark_summary.csv", index=False)

    print("\n===== FULL BENCHMARK SUMMARY =====")
    print(summary.to_string(index=False))

    return summary



# Statistical Significance Tests


def statistical_tests(df):
    rows = []

    for function_name in df["function"].unique():
        subset = df[df["function"] == function_name]

        rhcr = subset[subset["algorithm"] == "RHCR5"]["final_value"]

        for other_algorithm in ["Basic RHC", "Random Search", "Simulated Annealing"]:
            other = subset[subset["algorithm"] == other_algorithm]["final_value"]

            t_stat, p_val = ttest_ind(rhcr, other, equal_var=False)

            rows.append({
                "function": function_name,
                "comparison": f"RHCR5 vs {other_algorithm}",
                "t_statistic": t_stat,
                "p_value": p_val,
                "significant_at_0.05": p_val < 0.05
            })

    tests = pd.DataFrame(rows)
    tests.to_csv("statistical_tests.csv", index=False)

    print("\n===== STATISTICAL TESTS =====")
    print(tests.to_string(index=False))

    return tests



# Mean Convergence Curves


def pad_history(history, length):
    if len(history) >= length:
        return history[:length]

    return history + [history[-1]] * (length - len(history))


def mean_convergence_plot(function_name="Frog", num_seeds=50, budget=7203):
    config = FUNCTIONS[function_name]

    func = config["func"]
    bounds = config["bounds"]
    start = config["start"]
    z = config["z"]
    p = config["p"]

    histories = {
        "Basic RHC": [],
        "RHCR5": [],
        "Random Search": [],
        "Simulated Annealing": []
    }

    for seed in range(num_seeds):
        histories["Basic RHC"].append(
            pad_history(RHC(func, start, z, p, seed, bounds, budget=budget, track=True)["history"], budget)
        )

        histories["RHCR5"].append(
            pad_history(RHCR5(func, start, z, p, seed, bounds, budget=budget, track=True)["history"], budget)
        )

        histories["Random Search"].append(
            pad_history(random_search(func, seed, bounds, budget=budget, track=True)["history"], budget)
        )

        histories["Simulated Annealing"].append(
            pad_history(simulated_annealing(func, start, z, seed, bounds, budget=budget, track=True)["history"], budget)
        )

    plt.figure(figsize=(10, 5))

    for algorithm, runs in histories.items():
        mean_history = np.mean(np.array(runs), axis=0)
        plt.plot(mean_history, label=algorithm)

    plt.xlabel("Function Evaluations")
    plt.ylabel("Mean Best Objective Value So Far")
    plt.title(f"Mean Convergence Curve on {function_name} Function")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"mean_convergence_{function_name.lower()}.png", dpi=300)
    plt.show()



# Boxplots


def boxplot_results(df):
    for function_name in df["function"].unique():
        subset = df[df["function"] == function_name]
        algorithms = subset["algorithm"].unique()

        data = [
            subset[subset["algorithm"] == alg]["final_value"]
            for alg in algorithms
        ]

        plt.figure(figsize=(10, 5))
        plt.boxplot(data, tick_labels=algorithms)
        plt.ylabel("Final Objective Value")
        plt.title(f"Algorithm Stability on {function_name} Function")
        plt.xticks(rotation=20)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"boxplot_{function_name.lower()}.png", dpi=300)
        plt.show()



# Runtime Bar Chart


def runtime_chart(summary):
    for function_name in summary["function"].unique():
        subset = summary[summary["function"] == function_name]

        plt.figure(figsize=(10, 5))
        plt.bar(subset["algorithm"], subset["mean_runtime_seconds"])
        plt.ylabel("Average Runtime in Seconds")
        plt.title(f"Average Runtime by Algorithm on {function_name}")
        plt.xticks(rotation=20)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"runtime_{function_name.lower()}.png", dpi=300)
        plt.show()



# Success Rate Chart


def success_rate_chart(summary):
    for function_name in summary["function"].unique():
        subset = summary[summary["function"] == function_name]

        plt.figure(figsize=(10, 5))
        plt.bar(subset["algorithm"], subset["success_rate"])
        plt.ylabel("Success Rate (%)")
        plt.title(f"Success Rate by Algorithm on {function_name}")
        plt.xticks(rotation=20)
        plt.ylim(0, 100)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"success_rate_{function_name.lower()}.png", dpi=300)
        plt.show()



# Heatmap With RHCR5 Trajectory


def heatmap_with_trajectory(function_name="Frog", grid_size=300):
    config = FUNCTIONS[function_name]

    func = config["func"]
    bounds = config["bounds"]
    start = config["start"]
    z = config["z"]
    p = config["p"]

    seed = 321

    result = RHCR5(func, start, z, p, seed, bounds, track=True)
    path = result["path"]

    lower, upper = bounds

    x = np.linspace(lower, upper, grid_size)
    y = np.linspace(lower, upper, grid_size)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func((X[i, j], Y[i, j]))

    plt.figure(figsize=(9, 7))
    contour = plt.contourf(X, Y, Z, levels=80)
    plt.colorbar(contour, label=f"{function_name} Function Value")

    plt.plot(path[:, 0], path[:, 1], marker="o", linewidth=2, label="RHCR5 Search Path")
    plt.scatter(path[0, 0], path[0, 1], s=100, label="Start")
    plt.scatter(path[-1, 0], path[-1, 1], s=100, label="Final Solution")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{function_name} Function Landscape with RHCR5 Trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"heatmap_trajectory_{function_name.lower()}.png", dpi=300)
    plt.show()



# Best Results Printer


def print_best_results(df):
    print("\n===== BEST RESULTS BY FUNCTION =====")

    for function_name in df["function"].unique():
        subset = df[df["function"] == function_name]
        best = subset.loc[subset["final_value"].idxmin()]

        print(f"\nFunction: {function_name}")
        print(f"Best Algorithm: {best['algorithm']}")
        print(f"Seed: {int(best['seed'])}")
        print(f"Best Value: {best['final_value']:.12f}")
        print(f"Solution: ({best['x']:.6f}, {best['y']:.6f})")
        print(f"Calls: {best['calls']}")
        print(f"Runtime: {best['runtime_seconds']:.6f} seconds")
        print(f"Error: {best['error']:.6f}")
        print(f"Percent Error: {best['percent_error']:.6f}")



# Main


def main():
    df = benchmark_all(num_seeds=50, budget=7203)

    print_best_results(df)

    summary = create_summary(df)

    statistical_tests(df)

    boxplot_results(df)

    runtime_chart(summary)

    success_rate_chart(summary)

    mean_convergence_plot("Frog", num_seeds=50)
    mean_convergence_plot("Rastrigin", num_seeds=50)
    mean_convergence_plot("Ackley", num_seeds=50)

    heatmap_with_trajectory("Frog", grid_size=300)
    heatmap_with_trajectory("Rastrigin", grid_size=300)
    heatmap_with_trajectory("Ackley", grid_size=300)

    print("\nFiles created:")
    print("full_algorithm_benchmark_results.csv")
    print("full_algorithm_benchmark_summary.csv")
    print("statistical_tests.csv")
    print("boxplot_frog.png")
    print("boxplot_rastrigin.png")
    print("boxplot_ackley.png")
    print("runtime_frog.png")
    print("runtime_rastrigin.png")
    print("runtime_ackley.png")
    print("success_rate_frog.png")
    print("success_rate_rastrigin.png")
    print("success_rate_ackley.png")
    print("mean_convergence_frog.png")
    print("mean_convergence_rastrigin.png")
    print("mean_convergence_ackley.png")
    print("heatmap_trajectory_frog.png")
    print("heatmap_trajectory_rastrigin.png")
    print("heatmap_trajectory_ackley.png")


if __name__ == "__main__":
    main()