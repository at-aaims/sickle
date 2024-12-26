import numpy as np
import matplotlib.pyplot as plt

# Load the data from the numpy file
data = np.load('snapshots/interpolated.npz')

# Extract the 'x' and 'y' arrays
x = np.squeeze(data['x'][0])
y = np.squeeze(data['y'][0])

# Create a 2D plot
plt.plot(x, y, 'o-', label='Interpolated Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Plot of Interpolated Data')
plt.legend()
plt.grid(True)
plt.show()

