import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_wireframe_cube(ax, origin, size, color='black'):
    """Draws a wireframe cube given an origin and size."""
    x, y, z = origin
    s = size

    # Define the 8 vertices of the cube
    vertices = np.array([
        [x, y, z],
        [x + s, y, z],
        [x + s, y + s, z],
        [x, y + s, z],
        [x, y, z + s],
        [x + s, y, z + s],
        [x + s, y + s, z + s],
        [x, y + s, z + s]
    ])

    # Define the 12 edges of the cube
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for edge in edges:
        ax.plot3D(*zip(*vertices[list(edge)]), color=color)

# Create a single figure and 3D axis
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Draw the large cube
draw_wireframe_cube(ax, origin=(0, 0, 0), size=10)

# Define positions for the three smaller cubes inside the larger one
small_size = 3
positions = [(1, 1, 1), (6, 1, 1), (6, 1, 9)]

# Draw the three smaller cubes
for pos in positions:
    draw_wireframe_cube(ax, origin=pos, size=small_size)

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])

plt.show()

