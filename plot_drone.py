import numpy as np
from tensegrity_drone import TensegrityDrone

def main():
    drone = TensegrityDrone(plot=True, angles=[0, 0, 0],
                            p=[0, 0, 0],
                            barrier_loc=[0, 0.11, 0],
                            barrier_sidelength=[0.2, 0.01, 0.2],
                            barrier_orientation=[0, 0, 0])
    drone.plot_tensegrity_drone(axis=True)
    drone.save("free_body_diagram.eps", dpi=300, transparent=True, bbox_inches="tight")

if __name__ == '__main__':
    main()