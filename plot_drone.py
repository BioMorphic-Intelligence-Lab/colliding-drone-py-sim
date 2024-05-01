from tensegrity_drone import TensegrityDrone

def main():
    drone = TensegrityDrone()
    drone.plot_tensegrity_drone()
    drone.show()
    #drone.save("vis.png")

if __name__ == '__main__':
    main()