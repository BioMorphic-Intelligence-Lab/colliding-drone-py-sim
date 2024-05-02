import numpy as np
from tensegrity_drone import TensegrityDrone

def main():
    drone = TensegrityDrone(plot=True, angles=[0, np.pi/4, 0])
    drone.plot_tensegrity_drone()
    drone.show()

if __name__ == '__main__':
    main()