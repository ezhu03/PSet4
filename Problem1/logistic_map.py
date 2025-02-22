from matplotlib import pyplot as plt
import numpy as np
def logistic_fixed_points(r):
    """
    Returns the fixed points of the logistic map for a given r.
    The fixed points satisfy: x = r x (1-x).
    """
    # Always one fixed point at 0
    fixed_points = [0]
    
    # The second fixed point (if r != 0)
    if r != 0:
        fixed_points.append(1 - 1/r)
    return fixed_points

def derivative(r, x):
    """
    Computes the derivative of the logistic map at x.
    f'(x) = r (1 - 2x)
    """
    return r * (1 - 2 * x)

def stability_label(deriv):
    """
    Returns a label for stability based on the magnitude of the derivative.
    """
    if abs(deriv) < 1:
        return "stable"
    elif abs(deriv) > 1:
        return "unstable"
    else:
        return "marginal"

# Test for r = 1, 2, 3, 4
r_values = [1, 2, 3, 4]

for r in r_values:
    print(f"\nFor r = {r}:")
    fps = logistic_fixed_points(r)
    for x in fps:
        d = derivative(r, x)
        stability = stability_label(d)
        print(f"  Fixed point x = {x:.4f}, f'(x) = {d:.4f} -> {stability}")

def iterate_logistic_map(r, x0, threshold=1e-4, max_iter=100000):
    """
    Iterates the logistic map:
        x[n+1] = r * x[n] * (1 - x[n])
    until the change between iterations is less than threshold,
    or until max_iter iterations have been reached.
    
    Parameters:
        r         : growth parameter.
        x0        : initial condition.
        threshold : convergence threshold.
        max_iter  : maximum number of iterations.
        
    Returns:
        sequence : list of iterated values.
    """
    sequence = [x0]
    x = x0
    for i in range(max_iter):
        x_next = r * x * (1 - x)
        sequence.append(x_next)
        if abs(x_next - x) < threshold:
            # Convergence achieved.
            break
        x = x_next
    return sequence

# Parameters
x0 = 0.2
threshold = 1e-6
r_values = [2, 3, 3.5, 3.8, 4.0]

# Iterate for each r value
for r in r_values:
    seq = iterate_logistic_map(r, x0, threshold, max_iter=100000)
    print(f"r = {r}")
    if len(seq) >= 100000:
        print("  Did not converge after 100,000 iterations.")
    else:
        print(f"  Converged after {len(seq)-1} iterations to {seq[-1]:.6f}")
    # Optionally, uncomment the next line to see the full sequence:
    # print("  Sequence:", seq)
    print()

# List of r values and initial conditions to test
r_values = [2, 3, 3.5, 3.8, 4.0]
initial_conditions = [0.1,0.2, 0.3, 0.5]

# Number of iterations for the time series
num_iter = 100

# Create a figure with one subplot per r value
fig, axes = plt.subplots(len(r_values), 1, figsize=(10, 15), sharex=True)

for ax, r in zip(axes, r_values):
    for x0 in initial_conditions:
        # Compute the time series for a given r and initial condition
        series = iterate_logistic_map(r, x0, max_iter=num_iter)
        ax.plot(series, marker='o', markersize=3, label=f"x₀ = {x0}")
    
    ax.set_title(f"Logistic Map Time Series for r = {r}")
    ax.set_ylabel("xₙ")
    ax.legend()
    ax.grid(True)

axes[-1].set_xlabel("Iteration (n)")
plt.tight_layout()
plt.show()
plt.savefig("logistic_map_time_series.png")

# Additionally, print the final values after iteration to see if they are the same.
print("Final values after iteration:")
for r in r_values:
    print(f"\nr = {r}")
    for x0 in initial_conditions:
        series = iterate_logistic_map(r, x0, num_iter)
        print(f"  x₀ = {x0:0.1f} → x_final = {series[-1]:.6f}")

# Define parameters for the bifurcation diagram
x0 = 0.2            # initial condition
r_min = 0.0         # minimum r value
r_max = 4.0         # maximum r value
num_r = 40000        # number of r values to sample (adjust for resolution)
final_iterates = 8  # number of iterates at the end of each sequence to plot

# Prepare arrays for r and x values (for plotting)
r_list = []
x_list = []

# Loop over r values
r_values = np.linspace(r_min, r_max, num_r)
for r in r_values:
    sequence = iterate_logistic_map(r, x0, threshold=1e-4, max_iter=1000)
    # If the sequence converged early, use all available iterates.
    # Otherwise, use the last 'final_iterates' values.
    if len(sequence) < final_iterates:
        tail = sequence
    else:
        tail = sequence[-final_iterates:]
    # Append a point (r, x) for each of the final iterates.
    for x in tail:
        r_list.append(r)
        x_list.append(x)

# Create the bifurcation diagram
plt.figure(figsize=(10, 7))
plt.scatter(r_list, x_list, s=0.1, color='black')
plt.xlabel('$r$', fontsize=14)
plt.ylabel('$x$', fontsize=14)
plt.title('Bifurcation Diagram for the Logistic Map\n(initial condition $x_0=0.2$)', fontsize=16)
plt.grid(True)
plt.show()
plt.savefig("logistic_map_bifurcation.png")
# Create the bifurcation diagram
plt.figure(figsize=(10, 7))
plt.scatter(r_list, x_list, s=0.1, color='black')
plt.xlabel('$r$', fontsize=14)
plt.ylabel('$x$', fontsize=14)
plt.xlim((2.5,4))
plt.title('Bifurcation Diagram for the Logistic Map\n(initial condition $x_0=0.2$)', fontsize=16)
plt.grid(True)
plt.show()
plt.savefig("logistic_map_bifurcation_2.5.png")
f=open("bifurcation.txt","w")
f.write("up until r_1 = 1: dies out to 0\n")
f.write("from r_1 up until r_2 = 3: stable fixed point\n")
f.write("bifurcation at r_3 = 3\n")
f.write("second bifurcation at r_4 ≈ 3.45\n")
f.write("third bifurcation at r_5 ≈ 3.54\n")
f.write("chaos from r_6 ≈ 3.57\n")
f.write("small windows of stability between r_7 ≈ 3.83 and r_8 ≈ 3.85\n")
f.write("return to chaos after r_8 ≈ 3.85\n")
f.close()
print("bifurcation.txt created")

# Define the range of gamma values
gamma_values = np.linspace(0.5, 1.5, 200)

# Compute the first bifurcation r-value for each gamma:
# r_bifurcation = (gamma + 2) / gamma = 1 + 2/gamma
r_bifurcation = (gamma_values + 2) / gamma_values

# Plotting the first bifurcation point as a function of gamma
plt.figure(figsize=(8, 6))
plt.plot(gamma_values, r_bifurcation, 'b-', linewidth=2, label=r'$r_{\rm bifurcation} = 1 + \frac{2}{\gamma}$')
plt.xlabel(r'$\gamma$', fontsize=14)
plt.ylabel(r'$r_{\rm bifurcation}$', fontsize=14)
plt.title('First Bifurcation Point vs. $\gamma$', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
plt.savefig("bifurcation_gamma.png")
