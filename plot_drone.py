import numpy as np
from tensegrity_drone import TensegrityDrone

def main():
    drone = TensegrityDrone(plot=True, angles=[0, 0, 0],
                            p=[0, 0, 0])
    drone.plot_tensegrity_drone(axis=True)
    drone.show()

if __name__ == '__main__':
    main()