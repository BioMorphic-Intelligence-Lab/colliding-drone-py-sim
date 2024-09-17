import numpy as np
from scipy.spatial.transform import Rotation as R


def draw_curved_arrow(ax, center, radius, r, start, stop, **kwargs):
    angles = np.linspace(start, stop, 100)
    path = np.array(
        [r @ np.array([radius * np.sin(angles[i]), radius * np.cos(angles[i]), 0]) for i in range(len(angles))])
    path += center 

    ax.plot(path[:, 0], path[:, 1], path[:, 2], **kwargs)

    direction = r @ R.from_rotvec(angles[-1] * np.array([0, 0, 1])).as_matrix() @ np.array([-1, 0, 0])

    # Draw arrowhead
    ax.quiver(path[-1, 0], path[-1, 1], path[-1, 2],
              direction[0], direction[1], direction[2], length=0.001, normalize=True, 
                               arrow_length_ratio=5, **kwargs)
