# RHCR5: Benchmarking Stochastic Optimization Algorithms

## Overview

This project explores how different optimization algorithms search for high-quality solutions to difficult mathematical problems. In many real-world applications, such as artificial intelligence, machine learning, engineering design, and scientific computing, it is often necessary to find the "best" solution among a very large number of possibilities. This process is known as optimization.

The goal of this project is to compare several stochastic optimization algorithms on challenging benchmark functions that contain many local optima. A local optimum is a solution that appears to be the best in a nearby region but is not necessarily the best solution overall. These landscapes can be difficult to navigate because algorithms may become trapped in suboptimal regions.

Four optimization methods were evaluated:

* Basic Randomized Hill Climbing (RHC)
* Randomized Hill Climbing with Five-Stage Refinement (RHCR5)
* Random Search
* Simulated Annealing

The study focuses on comparing solution quality, reliability, runtime, and convergence behavior across multiple benchmark functions.

---

## What is RHCR5?

RHCR5 is a custom optimization algorithm developed for this project.

Traditional hill climbing searches for better solutions by repeatedly exploring nearby points and moving whenever an improvement is found. RHCR5 extends this idea by performing the search in five stages, each using a smaller neighborhood size than the previous stage.

```text
Stage 1: z
Stage 2: z / 10
Stage 3: z / 50
Stage 4: z / 250
Stage 5: z / 1000
```

The early stages encourage broad exploration of the search space, while the later stages focus on fine-tuning the best solution found so far. This approach combines exploration and refinement in a single optimization framework.

---

## Benchmark Functions

Three well-known optimization benchmark functions were used.

### Frog Function

A highly irregular function with many local minima and a large search domain. The Frog function is useful for testing whether an algorithm can locate high-quality solutions in a difficult landscape.

### Rastrigin Function

A classic optimization benchmark containing many regularly spaced local minima surrounding a known global optimum.

### Ackley Function

A widely used benchmark that combines a large search region with a narrow basin around the global optimum, making it challenging for many optimization methods.

---

## Experimental Design

The benchmark consisted of:

* 4 optimization algorithms
* 3 benchmark functions
* 50 random seeds per algorithm

This resulted in:

```text
600 total optimization trials.
```

Each algorithm was evaluated using:

* Best solution found
* Average solution quality
* Success rate
* Runtime
* Function evaluations
* Convergence behavior
* Statistical significance testing

---

## Key Results

### Best Solutions Found

| Function  | Best Algorithm | Best Value        |
| --------- | -------------- | ----------------- |
| Frog      | RHCR5          | -511.732881886610 |
| Rastrigin | RHCR5          | 0.000000030692    |
| Ackley    | RHCR5          | 0.000029645737    |

### Success Rates

| Function  | RHCR5 Success Rate |
| --------- | ------------------ |
| Frog      | 56%                |
| Rastrigin | 98%                |
| Ackley    | 100%               |

### Major Findings

* RHCR5 produced the best observed solution on all three benchmark functions.
* RHCR5 achieved a 100% success rate on the Ackley function.
* RHCR5 achieved a 98% success rate on the Rastrigin function.
* RHCR5 found a Frog-function solution within approximately 0.052% of the benchmark target value.
* Statistical testing showed RHCR5 significantly outperformed competing methods on the Ackley benchmark.

---

## Visualizations Generated

The project automatically generates several figures:

### Performance Comparison

* Boxplots comparing algorithm performance across random seeds
* Success-rate comparisons
* Runtime comparisons

### Convergence Analysis

* Mean convergence curves showing how quickly algorithms improve over time

### Search Trajectories

* Heatmaps of each benchmark function
* Visualizations of the RHCR5 search path through the objective landscape

---


## Skills Demonstrated

This project demonstrates skills in:

### Artificial Intelligence

* Local Search Algorithms
* Stochastic Optimization
* Metaheuristics
* Search Space Exploration

### Data Science

* Experimental Design
* Statistical Analysis
* Hypothesis Testing
* Data Visualization

### Software Engineering

* Algorithm Development
* Benchmarking Frameworks
* Performance Evaluation
* Reproducible Research


---

## Author

**Waleed Farrakh**

