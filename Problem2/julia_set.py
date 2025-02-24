import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours

def julia_set(xmin, xmax, ymin, ymax, width, height, c, max_iter):
    # Create grid of complex numbers
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    # Initialize arrays to store iteration counts and a mask for non-diverged points
    iteration_counts = np.zeros(Z.shape, dtype=int)
    mask = np.full(Z.shape, True, dtype=bool)
    
    # Iterate the function: z -> z^2 + c
    for i in range(max_iter):
        Z[mask] = Z[mask]**2 + c
        still_bounded = np.abs(Z) <= 2
        # Record the iteration at which divergence occurred
        iteration_counts[mask & ~still_bounded] = i
        mask &= still_bounded
    
    return X, Y, iteration_counts, mask

def graham_scan(data):
    def polar_angle(p1, p2):
        return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    # Find the point with the smallest y-coordinate
    y_min_ind = np.argmin(data[:, 1])
    y_min = data[y_min_ind]
    # Create a dictionary mapping each point to its polar angle with respect to y_min
    polar_dict = {}
    for point in data:
        polar_dict.update({tuple(point): polar_angle(y_min, point)})
    # Sort points by polar angle
    polar_dict = dict(sorted(polar_dict.items(), key=lambda item: item[1]))
    polar_points = [key for key in polar_dict]
    hull = []
    # Build the convex hull using a stack (Graham scan)
    for point in polar_points:
        while len(hull) > 1 and np.cross(
            np.array(hull[-1]) - np.array(hull[-2]),
            np.array(point) - np.array(hull[-1])
        ) <= 0:
            hull.pop()
        hull.append(point)
    
    return np.array(hull)
def hull_area(hull_closed):
    # Remove the duplicate last point if present.
    if np.allclose(hull_closed[0], hull_closed[-1]):
        hull = hull_closed[:-1]
    else:
        hull = hull_closed
    # Get x and y coordinates
    x = hull[:, 0]
    y = hull[:, 1]
    # Apply the shoelace formula
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area
'''def contour_area(contour):
    """
    Compute the area of a polygon given by an ordered list of vertices using the shoelace formula.
    The contour should be a closed polygon (first vertex repeated at the end is optional).
    """
    # If the first point is not repeated, we can use np.roll to handle wrap-around.
    x = contour[:, 0]
    y = contour[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area'''


# Parameters for the Julia set
width, height = 800, 800
xmin, xmax = -1.5, 1.5
ymin, ymax = -1, 1
c = complex(-0.7, 0.356)
max_iter = 256

# Generate the Julia set and retrieve the grid and mask
X, Y, julia, mask = julia_set(xmin, xmax, ymin, ymax, width, height, c, max_iter)
# Display the result using matplotlib
plt.figure(figsize=(8, 8))
plt.imshow(julia, extent=(xmin, xmax, ymin, ymax), cmap='viridis', origin='lower')
plt.colorbar(label='Iteration count')
plt.title('Julia Set for f(z)=z^2 + (-0.7+0.356i)')
plt.xlabel('Real axis')
plt.ylabel('Imaginary axis')
plt.show()
plt.savefig('julia_set.png')

# Extract points that did not diverge (inside the Julia set)
points_inside = np.column_stack((X[mask], Y[mask]))

# Compute the convex (complex) hull of these points using the provided function
hull = graham_scan(points_inside)

# Plot the Julia set and overlay the convex hull
plt.figure(figsize=(8, 8))
plt.imshow(julia, extent=(xmin, xmax, ymin, ymax), cmap='viridis', origin='lower')
plt.scatter(points_inside[:, 0], points_inside[:, 1], s=1, color='blue', alpha=1, label='Julia set points')

# Close the hull polygon by appending the first vertex at the end
hull_closed = np.vstack([hull, hull[0]])
plt.plot(hull_closed[:, 0], hull_closed[:, 1], color='cyan', linewidth=2, label='Convex Hull')

plt.colorbar(label='Iteration Count')
plt.title('Julia Set with Convex Hull Overlay')
plt.xlabel('Real Axis')
plt.ylabel('Imaginary Axis')
plt.legend()
plt.show()
plt.savefig('julia_set_convex_hull.png')

area_hull = hull_area(hull_closed)
print("Area enclosed by the hull:", area_hull)

contours = find_contours(mask.astype(float), 0.5)
if contours:
    # Define the shoelace formula as a function.
    def polygon_area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    max_area = 0
    main_contour = None
    
    # Iterate over each contour to compute its enclosed area.
    for contour in contours:
        # Convert contour coordinates from image (row, col) to (x, y)
        contour_x = xmin + (xmax - xmin) * contour[:, 1] / (width - 1)
        contour_y = ymin + (ymax - ymin) * contour[:, 0] / (height - 1)
        area = polygon_area(contour_x, contour_y)
        if area > max_area:
            max_area = area
            main_contour = contour
    
    if main_contour is not None:
        # Convert the selected main contour's coordinates for further use.
        contour_x = xmin + (xmax - xmin) * main_contour[:, 1] / (width - 1)
        contour_y = ymin + (ymax - ymin) * main_contour[:, 0] / (height - 1)
        print("Contour Enclosed Area:", max_area)
    else:
        print("No valid contour found.")
else:
    print("No contour found.")

def box_count(contour, eps, domain_origin):
    """
    For each box size in eps (an array), count how many boxes (on a grid starting at domain_origin)
    contain at least one point from the contour.
    """
    counts = []
    for eps_val in eps:
        # For each point in the contour, determine which box it falls in.
        indices = np.floor((contour - domain_origin) / eps_val).astype(int)
        unique_indices = np.unique(indices, axis=0)
        counts.append(len(unique_indices))
    return np.array(counts)

# --------------------------- BOX-COUNTING FOR FRACTAL DIMENSION ---------------------------
# We'll apply box counting to the main contour points.
# Define a set of box sizes (epsilon) spanning several decades.
domain_size = xmax - xmin  # domain length in x (assumed similar for y)
# We'll choose epsilons from the full domain size down to about 1/100 of the domain.
epsilons = np.logspace(np.log10(domain_size), np.log10(domain_size/100), num=20)

# The domain origin for the boxes is (xmin, ymin)
domain_origin = np.array([xmin, ymin])
counts = box_count(main_contour, epsilons, domain_origin)

# In box counting, the fractal dimension D is estimated via:
#    D = lim_(eps->0) [ log N(eps) / log(1/eps) ]
# We fit a line to log(1/eps) vs log(N(eps)).
log_inv_eps = np.log(1/epsilons)
log_counts = np.log(counts)
coeffs = np.polyfit(log_inv_eps, log_counts, 1)
fractal_dimension = coeffs[0]
print("Estimated fractal (box-counting) dimension:", fractal_dimension)

# --------------------------- PLOTTING ---------------------------
# Plot the log-log relationship with the linear fit.
plt.figure(figsize=(8,6))
plt.scatter(log_inv_eps, log_counts, label="Data")
plt.plot(log_inv_eps, np.polyval(coeffs, log_inv_eps), 'r-', 
         label="Fit: slope = {:.3f}".format(fractal_dimension))
plt.xlabel("log(1/epsilon)")
plt.ylabel("log(N(epsilon))")
plt.title("Box-counting Method for Fractal Dimension")
plt.legend()
plt.show()
